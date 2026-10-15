from __future__ import annotations

import gc
import math
from pathlib import Path


def cosine(a, b):
    if len(a) != len(b) or not a:
        return -1.0
    aa = sum(x * x for x in a)
    bb = sum(x * x for x in b)
    return sum(x * y for x, y in zip(a, b)) / max(1e-12, math.sqrt(aa * bb))


def diarize(config, wav, offset=0.0):
    import soundfile as sf
    import torch
    from pyannote.audio import Pipeline
    path = config["diarization_model"]
    if not Path(path).exists():
        raise ValueError("Local pyannote Community-1 assets are missing")
    pipeline = Pipeline.from_pretrained(path)
    if pipeline is None:
        raise ValueError("The local speaker model could not be opened")
    # Keep speaker analysis on CPU; it must not displace the accepted Sem voice.
    data, rate = sf.read(str(wav), dtype="float32", always_2d=True)
    audio = {"waveform": torch.from_numpy(data.mean(axis=1)).unsqueeze(0),
             "sample_rate": rate}
    output = pipeline(audio)
    annotation = output.speaker_diarization
    exclusive = output.exclusive_speaker_diarization
    embeddings = output.speaker_embeddings
    labels = annotation.labels()
    if embeddings is None or len(embeddings) != len(labels):
        raise ValueError("Diarization did not provide embeddings for persistent speaker identity")
    vectors = {str(label): [float(x) for x in embeddings[i]] for i, label in enumerate(labels)}
    turns = [{"start": float(t.start) + offset, "end": float(t.end) + offset,
              "speaker": str(s)} for t, _, s in annotation.itertracks(yield_label=True)]
    aligned = [{"start": float(t.start) + offset, "end": float(t.end) + offset,
                "speaker": str(s)} for t, _, s in exclusive.itertracks(yield_label=True)]
    del pipeline, output, audio, data
    gc.collect()
    return turns, aligned, vectors


class SpeakerBank:
    def __init__(self, store, job_id, config, voice_pool):
        self.store, self.job_id, self.config = store, job_id, config
        self.pool = voice_pool
        self.known = store.speakers(job_id)

    def bind(self, vectors):
        mapping, problems, used = {}, [], set()
        for label, vector in vectors.items():
            if not vector or not all(math.isfinite(x) for x in vector):
                mapping[label] = None
                problems.append({"kind": "speaker_embedding_invalid", "local_speaker": label})
                continue
            ranked = sorted(((cosine(vector, value["embedding"]), key)
                             for key, value in self.known.items() if value.get("embedding")),
                            reverse=True)
            best = ranked[0] if ranked else (-1, None)
            margin = best[0] - (ranked[1][0] if len(ranked) > 1 else -1)
            if best[0] >= self.config["speaker_similarity"]:
                if best[1] in used or margin < self.config["speaker_margin"]:
                    mapping[label] = None
                    problems.append({"kind": "speaker_ambiguous", "local_speaker": label,
                                     "candidates": ranked[:3]})
                    continue
                ident = best[1]
                old = self.known[ident]
                old["embedding"] = [(x * 4 + y) / 5 for x, y in zip(old["embedding"], vector)]
            else:
                ident = f"speaker-{len(self.known) + 1:03d}"
                if len(self.known) >= len(self.pool):
                    mapping[label] = None
                    problems.append({"kind": "voice_pool_exhausted", "local_speaker": label})
                    continue
                self.known[ident] = {"embedding": vector, "voice": self.pool[len(self.known)],
                                     "label": ident}
            used.add(ident)
            mapping[label] = ident
            self.store.put_speaker(self.job_id, ident, self.known[ident])
        return mapping, problems

    def voice(self, ident):
        value = self.known.get(ident)
        if value is None:
            raise ValueError("Resolve the speaker before synthesizing this line")
        return value["voice"]


def cue_speaker(cue, turns, mapping, minimum):
    duration = max(.001, cue["end"] - cue["start"])
    overlap = {}
    for turn in turns:
        value = max(0, min(cue["end"], turn["end"]) - max(cue["start"], turn["start"]))
        overlap[turn["speaker"]] = overlap.get(turn["speaker"], 0) + value
    ranked = sorted(overlap.items(), key=lambda pair: pair[1], reverse=True)
    if not ranked or ranked[0][1] / duration < minimum:
        return None, "insufficient speaker overlap"
    if len(ranked) > 1 and ranked[1][1] >= ranked[0][1] * .55:
        return None, "line covers more than one speaker; split or assign it"
    speaker = mapping.get(ranked[0][0])
    return speaker, None if speaker else "speaker identity is ambiguous"
