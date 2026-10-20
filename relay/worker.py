"""Owned, bounded audio worker. Invoked only when the Relay service receives work."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

from .config import Config
from .language import LanguageTools, iso, words, word_error
from .media import (RATE, CHANNELS, atomic_json, extract_audio, file_hash,
                    fingerprint, subtitle_cues)
from .process import ProcessOwner, choose_device, Cancelled
from .quality import separation_report
from .speakers import SpeakerBank, cue_speaker, diarize
from .speech import Speech
from .store import Store
from . import actor_beta_hook as _beta
from . import demucs_bed_parent as _bed
from . import actor_surround as _surround


def cue_id(cue):
    return hashlib.sha256(json.dumps(
        [round(cue["start"], 2), round(cue["end"], 2), cue["text"]],
        ensure_ascii=False).encode()).hexdigest()[:24]


def pcm_write(path, data, frames):
    import numpy as np
    import soundfile as sf
    if data.ndim == 1:
        data = np.repeat(data[:, None], CHANNELS, axis=1)
    if len(data) < frames:
        data = np.pad(data, ((0, frames - len(data)), (0, 0)))
    data = data[:frames]
    peak = float(np.max(np.abs(data))) if data.size else 0
    gain = min(1.0, .98 / max(.98, peak))
    temp = Path(str(path) + ".tmp.wav")
    sf.write(str(temp), data * gain, RATE, subtype="PCM_16")
    os.replace(temp, path)
    return gain


def context_seconds(config):
    """Observed context around each chunk; finite and non-negative.

    Natural-short eligibility already demands continuation-gap context at
    every analysis edge, so a small value only makes keeps rarer.
    """
    value = config["context_seconds"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("context_seconds must be a finite non-negative number of seconds")
    return float(value)


def legacy_refusal(config, store, job_id, spec):
    """Why the whole-window Demucs path must not process this job, or None.

    That path downmixes with -ac 2, removes every vocal and never reads
    kept-original decisions, so it must never touch cinematic actor work,
    independently of the engine fingerprint.
    """
    if spec.get("speech_engine", config["speech_engine"]) == "actor":
        return "A cinematic actor job cannot use the legacy separation path"
    if store.protected_originals(job_id, -1e18, 1e18):
        return "Job has kept-original decisions; the legacy separation path cannot honour them"
    if any(row["payload"].get("ledger_version") is not None
           for row in store.utterances(job_id, -1e18, 1e18)):
        return "Job has canonical actor takes; the legacy separation path cannot reuse them"
    return None


def process_chunk(config, store, job_id, index, owner, actor_pipeline=None):
    if config['speech_engine'] == 'actor':
        return process_actor_chunk(config,store,job_id,index,owner,actor_pipeline)
    import numpy as np
    import soundfile as sf
    started = time.monotonic()
    job = store.job(job_id)
    if not job or job["state"] == "cancelled":
        raise Cancelled("Job is cancelled")
    spec = job["payload"]
    refusal = legacy_refusal(config, store, job_id, spec)
    if refusal:
        raise ValueError(refusal)
    context = context_seconds(config)
    if fingerprint(spec["source"]) != spec["source_fingerprint"]:
        raise ValueError("Source media changed; create a new job")
    if config.engine_fingerprint() != spec["engine_fingerprint"]:
        raise ValueError("Engine settings changed; create a new job")
    if spec.get("subtitles") and fingerprint(spec["subtitles"]) != spec.get("subtitle_fingerprint"):
        raise ValueError("Subtitle file changed; create a new job")
    chunk = store.chunk(job_id, index)["payload"]
    begin, end = chunk["start"], chunk["end"]
    origin = max(0.0, begin - context)
    stop = min(spec["duration"], end + context)
    directory = config.cache / job_id / f"chunk-{index:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    revision = int(spec.get("revision", 1))
    issues, timing = [], {}

    def progress(stage):
        owner.check()
        current = store.job(job_id)
        if not current or current["state"] == "cancelled" or current["payload"].get("revision", 1) != revision:
            raise Cancelled("Job changed or was cancelled")
        store.event(job_id, "stage", chunk=index, stage=stage)

    def checkpoint(result, state):
        progress("saving-receipt")
        result.update({"revision": revision, "chunk": index, "start": begin, "end": end,
                       "elapsed_seconds": time.monotonic() - started,
                       "real_time_factor": (time.monotonic() - started) / max(.01, end - begin),
                       "source_fingerprint": spec["source_fingerprint"]})
        atomic_json(directory / "receipt.json", result)
        store.update_chunk(job_id, index, state, **result)
        store.event(job_id, "chunk", chunk=index, state=state,
                    elapsed_seconds=result["elapsed_seconds"])
        return result

    progress("extracting")
    original_path = directory / "original.wav"
    extract_audio(config, owner, spec["source"], spec["audio_index"], origin,
                  stop - origin, original_path)
    original, rate = sf.read(str(original_path), dtype="float32", always_2d=True)
    if rate != RATE or original.shape[1] != CHANNELS:
        raise ValueError("Decoded source format differs from the streaming contract")
    # Recognition, separation and translation remain CPU-only. Actor cloning
    # has its own separately admitted native GPU session.
    device = 'cpu'
    tools = LanguageTools(config, device)
    progress("recognizing-language")
    transcription, spoken = tools.speech(original_path)
    offset_transcription = [{**x, "start": x["start"] + origin, "end": x["end"] + origin}
                            for x in transcription]
    subtitles, subtitle_source = subtitle_cues(
        config, owner, spec["source"], spec.get("subtitles"), spec.get("subtitle_index"), discover=False)
    selected = [dict(x) for x in subtitles if x["end"] > origin and x["start"] < stop]
    cues = selected or offset_transcription
    cue_origin = "subtitles" if selected else "speech-recognition"
    for cue in cues:
        cue["id"] = cue_id(cue)
    overrides = spec.get("overrides", {})
    for cue in cues:
        override = overrides.get(cue["id"], {})
        if "source_text" in override:
            cue["text"] = str(override["source_text"]).strip()
        detected, probability, basis = tools.text_language(cue["text"], cue.get("language"))
        if len(cue["text"]) < 16 and not cue.get("language") and spoken["probability"] >= config["language_confidence"]:
            # A one-word name is not adequate evidence of a language change.
            detected, probability, basis = spoken["language"], spoken["probability"], "spoken-language-prior"
        cue["source_language"] = iso(override.get("source_language") or detected)
        cue["language_confidence"] = probability
        cue["language_basis"] = basis
        if probability < config["language_confidence"] and not override.get("source_language"):
            issues.append({"kind": "language_uncertain", "cue": cue["id"],
                           "detected": detected, "confidence": probability})

    frames = round(end * RATE) - round(begin * RATE)
    trim = round((begin - origin) * RATE)
    ready_path = directory / "english.wav"
    foreign_spoken = any(
        tools.text_language(x["text"])[0] != "en" and
        tools.text_language(x["text"])[1] > .9 and len(x["text"]) > 25
        for x in transcription)
    if (spoken["language"] == "en" and spoken["probability"] >= config["language_confidence"]
            and transcription and not foreign_spoken and not spec.get("force_redub")):
        pcm_write(ready_path, original[trim:trim + frames], frames)
        return checkpoint({"audio": str(ready_path), "sha256": file_hash(ready_path),
                           "mode": "original-English", "language": spoken,
                           "cues": [{**x, "id": cue_id(x), "english": x["text"]}
                                    for x in offset_transcription],
                           "quality": {"machine_checks_passed": True, "issues": [],
                                       "scope": "Detected existing English dialogue; no replacement required"}},
                          "ready")

    progress("identifying-speakers")
    turns, aligned, vectors = diarize(config, original_path, origin)
    if not cues:
        if turns:
            return checkpoint({"issues": [{"kind": "speech_without_transcript"}],
                               "original": str(original_path), "origin": origin,
                               "language": spoken, "cues": []}, "needs_review")
        pcm_write(ready_path, original[trim:trim + frames], frames)
        return checkpoint({"audio": str(ready_path), "sha256": file_hash(ready_path),
                           "mode": "no-dialogue", "language": spoken, "cues": [],
                           "quality": {"machine_checks_passed": True, "issues": []}}, "ready")

    speech = Speech(config, owner)
    bank = SpeakerBank(store, job_id, config, speech.pool())
    mapping, identity_problems = bank.bind(vectors)
    issues.extend(identity_problems)
    for cue in cues:
        assigned, problem = cue_speaker(cue, aligned, mapping, config["speaker_overlap"])
        override = overrides.get(cue["id"], {})
        cue["speaker"] = override.get("speaker") or assigned
        if cue["speaker"] not in bank.known:
            issues.append({"kind": "speaker_unresolved", "cue": cue["id"], "reason": problem})
        try:
            if "english_text" in override:
                cue["english"] = str(override["english_text"]).strip()
                cue["translation"] = "owner-correction"
            else:
                cue["english"], cue["translation"] = tools.english(
                    cue["text"], cue["source_language"])
            if not cue["english"]:
                raise ValueError("No English dialogue was produced")
        except Exception as error:
            issues.append({"kind": "translation_unavailable", "cue": cue["id"], "reason": str(error)})
            cue["english"] = ""
    if not selected:
        for cue in cues:
            if (cue.get("log_probability", 0) < -1.2 or cue.get("no_speech_probability", 0) > .6) and not overrides.get(cue["id"], {}).get("source_text"):
                issues.append({"kind": "uncertain_transcript", "cue": cue["id"]})

    progress("separating-dialogue")
    owner.run([sys.executable, "-m", "relay.separate", original_path, "--out-dir", directory,
               "--device", device, "--model", config["separation_model"],
               "--model-dir", config["separation_model_dir"]], timeout=1800)
    bed_normal = directory / "bed-48k.wav"
    owner.run([config["ffmpeg"], "-hide_banner", "-loglevel", "error", "-nostdin",
               "-i", directory / "bed.wav", "-ar", RATE, "-ac", CHANNELS, "-c:a",
               "pcm_s16le", "-y", bed_normal], timeout=120)
    bed, _ = sf.read(str(bed_normal), dtype="float32", always_2d=True)
    if len(bed) < len(original) - RATE // 10:
        raise ValueError("Separation output is shorter than its input")
    if len(bed) < len(original):
        bed = np.pad(bed, ((0, len(original) - len(bed)), (0, 0)))
    progress("checking-remaining-dialogue")
    bed_speech, _ = tools.speech(bed_normal, spoken["language"] if spoken["probability"] >= .65 else None)
    quality = separation_report(config, original, bed, RATE, cues, origin, bed_speech)
    issues.extend(quality["issues"])
    pre_render_errors = {"speaker_unresolved", "translation_unavailable", "language_uncertain",
                         "speaker_embedding_invalid", "voice_pool_exhausted", "uncertain_transcript"}
    if any(x["kind"] in pre_render_errors for x in issues):
        return checkpoint({"issues": issues, "quality": quality, "cues": cues, "origin": origin,
                           "original": str(original_path), "bed": str(bed_normal),
                           "language": spoken, "subtitle_source": subtitle_source}, "needs_review")

    progress("voicing-English")
    mixed = bed.copy()
    voice_cache = config.cache / job_id / "utterances"
    existing = store.utterances(job_id, origin, stop)
    rendered = []
    from difflib import SequenceMatcher
    for cue in cues:
        owner.check()
        # Reuse a canonical line crossing a processing boundary.
        prior = next((x["payload"] for x in existing
                      if x["payload"].get("revision") == revision
                      and x["payload"].get("speaker") == cue["speaker"]
                      and min(x["end"], cue["end"]) - max(x["start"], cue["start"]) >
                          .5 * min(x["end"] - x["start"], cue["end"] - cue["start"])
                      and SequenceMatcher(None, x["payload"].get("source_text", ""),
                                          cue["text"]).ratio() > .75), None)
        if prior and Path(prior["audio"]).is_file() and file_hash(prior["audio"]) == prior["sha256"]:
            line = prior
            if line.get("timing_overrun"):
                issues.append({"kind": "speech_timing_overrun", "cue": cue["id"]})
            if line.get("readback_word_error", 1) > .25:
                issues.append({"kind": "speech_readback_mismatch", "cue": cue["id"],
                               "reason": "Cached speech still requires listening review"})
        else:
            path, receipt = speech.render(cue["english"], bank.voice(cue["speaker"]),
                                          voice_cache, cue["end"] - cue["start"])
            line = {**receipt, "id": cue["id"], "speaker": cue["speaker"],
                    "audio": str(path), "source_text": cue["text"], "start": cue["start"],
                    "end": cue["end"], "revision": revision}
            if receipt["timing_overrun"]:
                issues.append({"kind": "speech_timing_overrun", "cue": cue["id"],
                               "seconds": receipt["seconds"], "window": cue["end"] - cue["start"]})
            heard, _ = tools.speech(path, "en")
            error_rate = word_error(cue["english"], " ".join(x["text"] for x in heard))
            line["readback_word_error"] = error_rate
            if error_rate > .25:
                issues.append({"kind": "speech_readback_mismatch", "cue": cue["id"],
                               "expected": cue["english"], "heard": " ".join(x["text"] for x in heard)})
            store.put_utterance(job_id, cue["id"], cue["start"], cue["end"], line)
        if any(x["id"] == line["id"] for x in rendered):
            continue
        samples, sample_rate = sf.read(line["audio"], dtype="float32", always_2d=True)
        if sample_rate != RATE:
            raise ValueError("Cached speech has an unexpected sample rate")
        start_frame = round((line["start"] - origin) * RATE)
        left = max(0, start_frame)
        source_left = max(0, -start_frame)
        count = min(len(mixed) - left, len(samples) - source_left)
        if count > 0:
            mixed[left:left + count] += samples[source_left:source_left + count].mean(axis=1)[:, None]
        rendered.append(line)
    gain = pcm_write(ready_path, mixed[trim:trim + frames], frames)
    quality["issues"] = issues
    quality["machine_checks_passed"] = not issues
    result = {"audio": str(ready_path), "sha256": file_hash(ready_path),
              "original": str(original_path), "bed": str(bed_normal), "origin": origin,
              "language": spoken, "subtitle_source": subtitle_source, "cue_origin": cue_origin,
              "cues": cues, "utterances": rendered, "quality": quality, "issues": issues,
              "mode": "English-dub", "output_gain": gain, "device": device}
    return checkpoint(result, "needs_review" if issues else "ready")


def final_chunk(spec,index):
    """True for the job's last chunk (index chunks_total-1), whose render range ends at the container duration."""
    total=spec.get('chunks_total')
    return isinstance(total,int) and not isinstance(total,bool) and total>0 and int(index)==total-1


def rendered_end(chunk,result,final):
    """End of the render range ActorPipeline used: chunk['end'], or on the final chunk only the exact decoded audio
    end it reported (render_start < report render_end < chunk end). Anything else keeps chunk['end']."""
    value=result.get('render_end') if final else None
    if isinstance(value,(int,float)) and not isinstance(value,bool) and chunk['start']<value<chunk['end']:
        return float(value)
    return chunk['end']


def process_actor_chunk(config,store,job_id,index,owner,pipeline=None):
    """Keep the original channel layout until cinematic dialogue replacement."""
    from .actor_pipeline import ActorPipeline
    job=store.job(job_id)
    if not job or job['state']=='cancelled':raise Cancelled('Job is cancelled')
    spec=job['payload'];chunk=store.chunk(job_id,index)['payload']
    # Jobs created before the engine was recorded fall back to the fingerprint check.
    if spec.get('speech_engine','actor')!='actor':
        raise ValueError('This job was created for another speech engine; create a new job')
    context=context_seconds(config)
    if fingerprint(spec['source'])!=spec['source_fingerprint'] or config.engine_fingerprint()!=spec['engine_fingerprint']:
        raise ValueError('Source or engine changed; create a new job')
    if spec.get('subtitles') and fingerprint(spec['subtitles'])!=spec.get('subtitle_fingerprint'):
        raise ValueError('Subtitles changed; create a new job')
    origin=max(0,chunk['start']-context)
    end=min(spec['duration'],chunk['end']+context)
    directory=config.cache/job_id/f'chunk-{index:06d}'
    directory.mkdir(parents=True,exist_ok=True)
    # The ffprobe stream the job selected supplies the source layout evidence;
    # ActorPipeline also checks the decoded WAV header against it.
    audio_stream=next((s for s in spec['media']['streams'] if s.get('index')==spec['audio_index']),None)
    if audio_stream is None or audio_stream.get('codec_type','audio')!='audio':
        raise ValueError('The selected audio stream is missing from the probed media')
    channels=int(audio_stream.get('channels',0))
    if channels not in (1,2,6) or (channels==6 and audio_stream.get('channel_layout') not in ('5.1','5.1(side)')):
        raise ValueError('The actor path requires a known mono, stereo or 5.1 layout')
    original=directory/'original-layout.wav'
    store.event(job_id,'stage',chunk=index,stage='cinematic-actor-dubbing')
    # Preserve the selected stream's own channel order: a forced -ac 6 would
    # remap 5.1(side) to the 5.1 default. extract_audio verifies the WAV mask.
    layout=audio_stream['channel_layout'] if channels==6 else {1:'mono',2:'stereo'}[channels]
    extract_audio(config,owner,spec['source'],spec['audio_index'],origin,end-origin,original,
                  channels=channels,preserve_layout=layout)
    # Window-bounded captions with provenance (captions.subtitle_source); caption_source keeps its legacy string.
    from .embedded_subtitles import chunk_captions, spec_choice
    captions,caption_source=chunk_captions(config,owner,spec,origin,end,directory)
    # firstpass-runtime-glue-v2: B caption targets receive the job's subtitle choice; flag off keeps the old call.
    _caption=(config.data if hasattr(config,'data') else config).get('actor_beta')
    caption_kwargs=({'subtitle_choice':spec_choice(spec)} if isinstance(_caption,dict) and _caption.get('enabled') is True
                    and _caption.get('caption_targets') is True else {})
    owned=pipeline is None
    if owned:pipeline=ActorPipeline(config)
    # The movie's last chunk: ActorPipeline may end its render range at the exact decoded audio end.
    final=final_chunk(spec,index)
    try:
        result=pipeline.process(original,origin,captions,directory,store,job_id,owner,
                                render_start=chunk['start'],render_end=chunk['end'],revision=spec.get('revision',1),
                                source_channel_layout=dict(audio_stream),
                                source_identity={'source_fingerprint':spec['source_fingerprint'],
                                                 'audio_index':spec['audio_index']},final_window=final,**caption_kwargs)
    finally:
        if owned:pipeline.close()
    owner.check()
    current=store.job(job_id)
    if current['state']=='cancelled' or current['payload'].get('revision',1)!=spec.get('revision',1):
        raise Cancelled('Job changed during actor processing')
    # The chunk's recorded timeline is the render range the pipeline actually used: chunk['end'], or on the final
    # chunk only, the exact decoded audio end it reported. english.wav, the receipt and the store agree on it.
    chunk_end=rendered_end(chunk,result,final)
    result=_beta.unattended_window(config,result,original)
    if result.get('audio'):
        ready=directory/'english.wav'
        owner.run([config['ffmpeg'],'-hide_banner','-loglevel','error','-nostdin','-ss',str(chunk['start']-origin),
            '-i',result['audio'],'-t',str(chunk_end-chunk['start']),'-ar',RATE,'-ac',CHANNELS,'-c:a','pcm_s16le','-y',ready])
        surround=Path(result['audio']).with_name(_surround.SIDECAR)           # before `audio` points at english.wav
        result.update(audio=str(ready),sha256=file_hash(ready))
        # The 5.1 mix of the same 30 s as english.wav, bound to it by hash (relay/actor_surround.py). Only the mix this render's own
        # report names is published; anything else is removed.
        _surround.publish(surround,directory,result.get('surround_mix'),result['sha256'],
            lambda source,target:owner.run([config['ffmpeg'],'-hide_banner','-loglevel','error','-nostdin','-ss',str(chunk['start']-origin),
                '-i',str(source),'-t',str(chunk_end-chunk['start']),'-ar',RATE,'-ac','6','-c:a','flac','-sample_fmt','s32','-y',target]))
        # firstpass-runtime-glue-v1: never Demucs twice; stdlib check before the post-ready cleanup.
        _demucs=(config.data if hasattr(config,'data') else config).get('actor_beta')
        if (isinstance(_demucs,dict) and _demucs.get('enabled') is True and _demucs.get('demucs_first_pass') is True
                and _demucs.get('demucs_mix_bed') is True):
            raise ValueError('actor_beta.demucs_first_pass and demucs_mix_bed are mutually exclusive')
        # Opt-in actor_beta.demucs_mix_bed: owned child remixes the bed only; flag off returns result unchanged.
        result=_bed.after_ready(config,result,original,directory,ready,owner,store=store,job_id=job_id,
                                revision=spec.get('revision',1))
    result.update(revision=spec.get('revision',1),chunk=index,start=chunk['start'],end=chunk_end,
                  source_fingerprint=spec['source_fingerprint'],subtitle_source=caption_source,
                  mode=result.get('mode','actor-English-candidate'))
    # Construction does not establish dialogue removal or actor fidelity.
    passthrough=result.get('state')=='ready' and result.get('mode')=='no-dialogue'
    result['quality']={'machine_checks_passed':passthrough,'issues':result['issues'],
                       'human_listening_acceptance':False,'scope':'Cinematic actor candidate pending real-scene acceptance'}
    result['pipeline_state']=result.pop('state','needs_review')
    state='ready' if passthrough else 'waiting_resources' if result['pipeline_state']=='waiting_resources' else 'needs_review'
    state=_beta.unattended_state(config,result,state)
    atomic_json(directory/'receipt.json',result)
    store.update_chunk(job_id,index,state,**result)
    store.event(job_id,'chunk',chunk=index,state=state)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--chunk", type=int, required=True)
    args = parser.parse_args()
    config = Config(args.config)
    import torch
    torch.set_num_threads(int(config["cpu_threads"]))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    owner = ProcessOwner()
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: owner.cancel())
    try:
        process_chunk(config, Store(config.state), args.job, args.chunk, owner)
    except Cancelled:
        raise SystemExit(3)
    except Exception as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
