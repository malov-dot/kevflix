"""Resident cinematic dubbing candidate.

Model analysis (ASR, translation, diarization, speaker embeddings) runs through the injected AnalysisService
(set_analysis_service): GPU models in its lease-owned child, never in this process. Local CPU analysis exists only
as explicit opt-in (local_cpu_analysis=True); it is never a silent fallback. Raw inference is cached once, by
content and exact model identity, in the AnalysisService's GPU child, so a speaker rebinding re-runs only the cheap
plan here.

Unusable references, overlap, timing, or transcript evidence blocks publication.
The beta lane clones each line from its own speech; actor identity is diagnostic.
This path never substitutes a stock voice or overlays unresolved original speech.
"""
from __future__ import annotations
from pathlib import Path
import copy
import hashlib
import json
import math
import os
import re
import time
from .actor_render import ActorRenderer
from . import actor_beta_hook as _beta
from .beta_policy_validator import validate_policy_with_beta_fallback as _validate_policy
from . import subtitle_runtime as _subs
from . import dialogue_units as _units
from .cinematic import CinematicSeparator,replacement_envelope,replace_dialogue,actor_mix_levels,CHECKPOINT_SHA256,take_removal_intervals,merge_intervals
from . import take_fit as _fit
from . import take_deess as _deess
from .nonverbal import nonverbal_reason,sparse_speech_reason,audio_features as tone_features,translation_artifact_reason
from . import audio_evidence as _evidence
from .dialogue import plan_dialogue,overlap,caption_text,is_no_speech
from . import beta_caption_flow as _ctf, beta_first_pass_flow as _fpf   # stdlib-only at import; flags default off
from . import beta_dialogue_ownership as _own   # stdlib-only; used only with actor_beta.caption_targets
from . import beta_line_ledger as _lines        # stdlib-only (sqlite3); used only with actor_beta.caption_targets
from .language import iso,aligned_caption_style,word_error
from .media import atomic_json,file_hash,resolve_channel_layout,wav_format,WAV_MASK_LAYOUTS
from .speakers import SpeakerBank,cosine,speaker_units
from .movie_faults import MovieConfigurationError,MovieContentFallback
from .timing import schedule_lines
from .resource_budget import ResourceYield
from .actor_reference import make_reference,validate_reference_file,validate_reference_audio
from .dialogue_ledger import (DialogueLedger,ProtectionLedger,add_utterance,merge_spans,
                              persisted_protection_issues,protection_rows,protection_scope,take_span)
from .verified_speech import critical_readback_changes
from .asr_rescue import rescue_windows,merge_rescue
from .dubbing_timing import plan_dubbing_timing
from .original_audio_policy import plan_original_audio,protect_dialogue_mix,original_protection_masks
from .prepared_actor import validate_original_policy,ORIGINAL_POLICY_VERSION,NATURAL_POLICY_VERSION

# Owner decision 2026-10-19: original speech is removed ONLY while someone is actually speaking, never the pauses,
# whispers, steps or music around it, and a dub that is a little too long is fitted into its slot (atempo <= 1.35).
SPEECH_ONLY_REMOVAL_VERSION='speech-only-removal-v1'
TAKE_FIT_MAX_SPEED=1.35
# A line at least this long (seconds) is cloned from its own audio (its own loudness/delivery); shorter lines use the bank reference.
PREFER_OWN_REFERENCE_SECONDS=1.5


def relax_solo_overlap(cues,issues,all_cues,tolerance=.12):
    """An 'overlapping dialogue' flag needs a second TRANSCRIBED line spoken at the same time. When the diarizer reports a
    second voice but no other cue overlaps this one, there is nothing to confuse: the line is dubbed (the removal is
    speech-only, so the unreferenced second voice goes with the stem). Returns the ids that were relaxed."""
    relaxed=[]
    for cue in cues:
        if not (cue.get('overlapped') or cue.get('turn_overlap') is True):continue
        if any(other is not cue and other.get('id')!=cue.get('id') and overlap(other,cue)>tolerance for other in all_cues):continue
        cue['overlapped']=False;cue.pop('turn_overlap',None);cue['overlap_relaxed']='no_second_transcribed_line'
        relaxed.append(cue['id'])
    if relaxed:issues[:]=[i for i in issues if not (i.get('kind')=='overlapping_dialogue' and i.get('cue') in relaxed)]
    return relaxed


def overlap_policy(config):
    """Tunable dialogue-quality thresholds (actor_beta settings; every default reproduces the earlier fixed behaviour).
    overlap_word_fraction: a word is 'uncertain overlap' when a second speaker covers more than this share of it.
    overlap_span_seconds / overlap_span_fraction: a cue is 'overlapped' when one shared span exceeds
    max(seconds, fraction * cue length). unclear_min_probability / unclear_word_floor: ASR confidence for 'unclear'."""
    beta=_beta.beta_settings(config) or {}
    def number(key,default,low,high):
        try:value=float(beta.get(key,default))
        except (TypeError,ValueError):return default
        return value if math.isfinite(value) and low<=value<=high else default
    return {'word_fraction':number('overlap_word_fraction',.25,.05,1.0),
            'span_seconds':number('overlap_span_seconds',.12,0.0,5.0),
            'span_fraction':number('overlap_span_fraction',0.0,0.0,1.0),
            'min_probability':number('unclear_min_probability',.65,.1,.95),
            'word_floor':number('unclear_word_floor',.5,.05,.9),
            'bank_min':number('bank_reference_min_seconds',BANK_REFERENCE_MIN_SECONDS,1.0,10.0),
            'fit_max_speed':number('take_fit_max_speed',TAKE_FIT_MAX_SPEED,1.0,1.6),
            'solo_relax':beta.get('overlap_solo_relax') is True,
            'fragment_seconds':number('merge_short_fragment_seconds',0.0,0.0,3.0),
            'low_word_share':number('unclear_low_word_share',0.0,0.0,0.6),
            'extend_merged':beta.get('reference_extend_merged_labels') is True,
            'borrow_reference':beta.get('borrow_reference') is True,
            'concise_retranslation':beta.get('concise_retranslation') is True,
            'language_correction':beta.get('language_correction') is True,
            'nonverbal_audio':beta.get('nonverbal_audio') is True,
            'reject_unintelligible':beta.get('reject_unintelligible') is True,
            'deess_takes':beta.get('deess_takes') is True}


REFERENCE_DONOR_MAX_GAP_SECONDS=36.
REFERENCE_DONOR_PITCH_SEMITONES=5.      # a donor whose median pitch is further than this from the line's is a different-sounding voice


def choose_reference_donors(cues,refused_ids,usable,speech_regions,*,max_gap=REFERENCE_DONOR_MAX_GAP_SECONDS):
    """Owner decision 2026-10-19: every spoken line is dubbed. A line whose own audio cannot be a reference (too short,
    overlapped by laughter, no usable embedding) borrows the reference of another cue of this window that already has a
    usable voice (usable(cue) -> bool). Preference: a similar-sounding voice (median pitch within
    REFERENCE_DONOR_PITCH_SEMITONES when both are known), then the same dominant raw diarization label, then the nearest
    in time, then the longer cue. A donor further than max_gap seconds is never used. Returns {cue_id: {'donor','same_raw_label',
    'gap_seconds'}}; a refused cue with no donor is absent (it keeps its original audio)."""
    def span(cue):return float(cue.get('speech_start') or cue['start']),float(cue.get('speech_end') or cue['end'])
    def label(cue):
        try:return dominant_reference_label(cue,speech_regions)
        except ValueError:return None
    donors=[c for c in cues if c['id'] not in refused_ids and not (c.get('overlapped') or c.get('turn_overlap') is True)
            and usable(c)]
    picks={}
    for cue in cues:
        if cue['id'] not in refused_ids:continue
        mine=label(cue);begin,stop=span(cue);ranked=[]
        for donor in donors:
            x,y=span(donor);gap=max(0.,x-stop,begin-y)
            if gap>max_gap:continue
            same=mine is not None and label(donor)==mine
            f1,f2=cue.get('f0_median_hz'),donor.get('f0_median_hz')
            far=bool(f1 and f2 and abs(12*math.log2(float(f1)/float(f2)))>REFERENCE_DONOR_PITCH_SEMITONES)
            ranked.append((1 if far else 0,0 if same else 1,round(gap,6),-(y-x),donor['id'],same,donor))
        if ranked:
            best=min(ranked,key=lambda item:item[:5])
            picks[cue['id']]={'donor':best[6],'same_raw_label':best[5],'gap_seconds':round(best[2],3),'pitch_mismatch':bool(best[0])}
    return picks


def vox_options(config):
    """Extra Vox render options. actor_beta.vox_reference_only (default off): clone from the reference AUDIO alone and never
    prompt Vox with the reference's source-language transcript (measured: the transcript prompt stretches English takes)."""
    return {'reference_only':True} if (_beta.beta_settings(config) or {}).get('vox_reference_only') is True else {}


def prefer_own_seconds(config):
    """Cue length from which a line is cloned from its own audio (actor_beta.prefer_own_reference_seconds, validated)."""
    value=(_beta.beta_settings(config) or {}).get('prefer_own_reference_seconds',PREFER_OWN_REFERENCE_SECONDS)
    try:value=float(value)
    except (TypeError,ValueError):return PREFER_OWN_REFERENCE_SECONDS
    return value if math.isfinite(value) and value>=1.0 else PREFER_OWN_REFERENCE_SECONDS


def fit_rendered_take(path,receipt,window,*,ffmpeg,runner=None,max_speed=TAKE_FIT_MAX_SPEED):
    """Owner decision 2026-10-19: a take that runs past its slot is fitted INTO it with one pitch-preserving atempo
    (<= TAKE_FIT_MAX_SPEED) instead of losing the whole cue to original audio. The fitted take is a NEW file named by the
    unfitted take's sha256 and the exact factor, with its own sha256 and receipt (fit receipt, unfitted length and hash
    kept), so claims, private copies, readback, the ledger and the mix see one consistent take. A take that fits, cannot
    be fitted (rate/channel mismatch, beyond the maximum speed) or whose fit fails (ffmpeg error, unreadable output) is
    returned UNCHANGED with the reason in receipt['fit']; the scheduler then reports its overrun for that cue only.
    Cancellation, resource waits and timeouts always propagate."""
    import soundfile as sf
    from .process import Cancelled
    seconds=float(receipt.get('seconds') or 0)
    if window<=0 or seconds<=window:return path,receipt
    if seconds/window>max_speed*1.02:
        return path,dict(receipt,fit={'applied':False,'reason':'beyond_max_speed','needed':round(seconds/window,3)})
    try:
        samples,take_rate=sf.read(str(path),dtype='float32')
        if take_rate!=48000 or samples.ndim!=1:
            return path,dict(receipt,fit={'applied':False,'reason':'unsupported_take_format'})
        fitted,fit=_fit.fit_take(samples,take_rate,window,max_speed=max_speed,ffmpeg=ffmpeg,runner=runner)
        if not fit.get('applied'):return path,dict(receipt,fit=fit)
        unfitted=str(receipt.get('sha256') or file_hash(Path(path)))
        target=Path(path).with_name('%s-fit-%s-%04d.wav'%(Path(path).stem[:48],unfitted[:16],round(fit['factor']*1000)))
        temp=target.with_name(target.name+'.%d.tmp'%os.getpid())
        try:
            # libsndfile stamps a wall-clock PEAK chunk, so identical samples written a second apart differ in bytes. Reuse an
            # existing fit whenever its PCM is identical, and only write (atomically) when it is not there yet.
            reuse=False
            if target.exists():
                import numpy as np
                try:
                    existing,existing_rate=sf.read(str(target),dtype='float32')
                    reuse=existing_rate==take_rate and existing.shape==fitted.shape and bool(np.array_equal(existing,fitted))
                except (RuntimeError,ValueError,OSError):reuse=False       # unreadable stale file: replace it
            if not reuse:
                sf.write(str(temp),fitted,take_rate,subtype='FLOAT',format='WAV')
                os.replace(temp,target)
        finally:
            if temp.exists():temp.unlink()
        new_seconds=len(fitted)/take_rate
        fitted_receipt=dict(receipt,seconds=new_seconds,sha256=file_hash(target),timing_overrun=new_seconds>window,
                            fit=fit,unfitted_seconds=seconds,unfitted_sha256=unfitted)
        if 'audio' in receipt:fitted_receipt['audio']=str(target)
        return target,fitted_receipt
    except (Cancelled,ResourceYield,TimeoutError):raise
    except (ValueError,RuntimeError,OSError) as error:
        return path,dict(receipt,fit={'applied':False,'reason':'fit_failed','error':'%s: %s'%(type(error).__name__,str(error)[:200])})


def deess_rendered_take(path,receipt):
    """Lower the hot fricative bursts of a rendered take (relay/take_deess.py: Vox \"s/sh/f\" bursts are 5 dB hotter than the
    actors' own and read as static when played loudly). Same contract as fit_rendered_take: the result is a NEW file named by
    the source take's sha256, with its own sha256 and receipt (the de-ess receipt and the earlier sha256 kept), so claims,
    private copies, readback, the ledger and the mix see one consistent take. A take with no hot burst, a format the
    de-esser does not handle, or any failure is returned UNCHANGED (the reason is in receipt['deess'])."""
    import soundfile as sf
    try:
        samples,take_rate=sf.read(str(path),dtype='float32')
        shaped,info=_deess.deess(samples,take_rate)
        if not info.get('applied'):return path,dict(receipt,deess=info)
        source=str(receipt.get('sha256') or file_hash(Path(path)))
        target=Path(path).with_name('%s-ds-%s.wav'%(Path(path).stem[:48],source[:16]))
        temp=target.with_name(target.name+'.%d.tmp'%os.getpid())
        try:
            reuse=False
            if target.exists():
                import numpy as np
                try:
                    existing,existing_rate=sf.read(str(target),dtype='float32')
                    reuse=existing_rate==take_rate and existing.shape==shaped.shape and bool(np.array_equal(existing,shaped))
                except (RuntimeError,ValueError,OSError):reuse=False
            if not reuse:
                sf.write(str(temp),shaped,take_rate,subtype='FLOAT',format='WAV')
                os.replace(temp,target)
        finally:
            if temp.exists():temp.unlink()
        shaped_receipt=dict(receipt,sha256=file_hash(target),deess=info,pre_deess_sha256=source)
        if 'audio' in receipt:shaped_receipt['audio']=str(target)
        return target,shaped_receipt
    except (ValueError,RuntimeError,OSError) as error:
        return path,dict(receipt,deess={'version':_deess.VERSION,'applied':False,'reason':'deess_failed','error':'%s: %s'%(type(error).__name__,str(error)[:200])})


# Owner decision 2026-10-19: Vox take length AND clarity vary with the sampling seed (the same words ran 3.8 s and read back
# perfectly on one seed, 1.3 s and read back as other words on another), so a take that does not fit its slot or does not
# read back clean is re-rendered on these seeds before the cue is given up.
TAKE_RETRY_SEEDS=(7,1234,2024)
READBACK_ORDINARY_ERROR=.25
PREFERRED_FIT_FACTOR=1.30      # a take sped up more than this is a poorer take: other seeds are tried before accepting it
UNINTELLIGIBLE_ERROR=1.0       # readback word error at or above this: the take says other words altogether
UNINTELLIGIBLE_SIMILARITY=.5   # ... and its letters resemble the intended words less than this


CONCISE_WORDS_PER_SECOND=3.4
CONCISE_SEEDS=(42,7,1234)


TRANSLATION_RETRY_CODES=frozenset({'translation_truncated','translation_empty'})

# Language of each recognized segment (owner decision 2026-10-20: lines already in English are never dubbed, and Russian
# speech must never be mistaken for English). A window is recognized in ONE language; a window with an English scene
# makes Whisper write the Russian speech around it down as English words (and the reverse). The spoken language of each
# segment is therefore read from the audio itself, and a segment whose audio language differs with confidence is decoded
# again in that language.
LID_MIN_SECONDS=.6
LID_MIN_PROBABILITY=.85
LID_PAD_SECONDS=.1
# Languages a segment may be re-decoded into: a short Russian aside was once identified as Chinese (p .85) and replaced by
# '不客气'. The detector's answer must be a language a film of this kind plausibly speaks, and the new decoding must look at
# least as confident as the one it replaces.
LID_ALLOWED=frozenset(('en','ru','uk','be','bg','sr','pl','cs','sk','es','pt','fr','it','de','nl','ro','ca','tr','el','hu','sv','da','no','fi'))
LID_MAX_CONFIDENCE_DROP=.05
# A re-decoding must also be a confident reading in its own right: two clean Spanish lines of a Spanish-language test film became Turkish
# ('Elbette. Elbette.') and English ('But hey, that's you.') at word confidence .56-.59, while the corrections that were right
# read .72 and above. Replacements that carry no word confidences (no word timestamps) are judged by the drop rule alone.
LID_MIN_REPLACEMENT_CONFIDENCE=.70


def shift_segments(segments,offset):
    """Copies of recognizer segments with every time moved by `offset` seconds."""
    return [{**seg,'start':float(seg['start'])+offset,'end':float(seg['end'])+offset,
             'words':[{**word,'start':float(word['start'])+offset,'end':float(word['end'])+offset} for word in seg.get('words') or []]}
            for seg in segments]


def mean_word_probability(segments):
    """Mean recognizer confidence over every word of the given segments (0 when there are none)."""
    values=[float(word.get('probability',0.)) for seg in segments for word in seg.get('words') or []]
    return sum(values)/len(values) if values else 0.


def apply_language_corrections(segments,ids,redecode,window_language,*,min_probability=LID_MIN_PROBABILITY):
    """segments: recognizer segments (seconds from the window origin); ids[i]: {'language','probability'} of segment i's
    audio or None; redecode(i, language) -> replacement segments or None. Returns (segments, corrections, spans) where
    spans are (start, end, language) of every re-decoded stretch."""
    out,corrections,spans=[],[],[]
    for index,segment in enumerate(segments):
        ident=ids[index] if index<len(ids) else None
        found=str((ident or {}).get('language') or '')
        probability=float((ident or {}).get('probability') or 0.)
        if ident and found in LID_ALLOWED and found!=window_language and probability>=min_probability:
            replacement=redecode(index,found)
            if (replacement and mean_word_probability(replacement)>=mean_word_probability([segment])-LID_MAX_CONFIDENCE_DROP
                    and (not any(item.get('words') for item in replacement)
                         or mean_word_probability(replacement)>=LID_MIN_REPLACEMENT_CONFIDENCE)):
                out.extend(replacement)
                corrections.append({'start':segment['start'],'end':segment['end'],'window_language':window_language,
                                    'detected':found,'probability':round(probability,3),'before':segment.get('text'),
                                    'after':' '.join(str(item.get('text','')) for item in replacement)})
                spans.append((min(item['start'] for item in replacement),max(item['end'] for item in replacement),found))
                continue
        out.append(segment)
    out.sort(key=lambda item:(item['start'],item['end']))
    return out,corrections,spans


cue_script_language='ru'


def cue_language(cue,spans,origin,default):
    """(language, from_audio): that of the re-decoded stretch covering most of the cue (confirmed by its audio), else the
    window's (not confirmed)."""
    begin,stop=float(cue['start'])-origin,float(cue['end'])-origin
    best,best_overlap,confirmed=default,0.,False
    letters=[ch for ch in str(cue.get('text') or '') if ch.isalpha()]
    if letters and default=='en' and sum(1 for ch in letters if 'Ѐ'<=ch<='ӿ')/len(letters)>=.7:
        best=cue_script_language   # Cyrillic words in a window labelled English are Russian, whatever the label says
    for a,b,found in spans:
        covered=max(0.,min(stop,b)-max(begin,a))
        if covered>best_overlap and covered>=.5*max(stop-begin,1e-6):best,best_overlap,confirmed=found,covered,True
    return best,confirmed


def nonverbal_evidence(cue_speech,rate_in,events):
    """Acoustic features of one cue's ORIGINAL vocals (any rate, mono or multichannel) merged with the pitch-recitation
    features the chant rule reads; {} when the excerpt is too short."""
    x=_evidence.to_mono_16k(cue_speech,rate_in)
    features=_evidence.cue_features(x,16000)
    if features:
        extra=tone_features(x,16000)
        if extra:features={**features,**extra}
    return features


def text_similarity(expected,heard):
    """0..1 resemblance of the letters of two texts (case, spacing and punctuation ignored)."""
    import difflib
    def plain(text):return re.sub(r'[\W_]+','',str(text).casefold())
    a,b=plain(expected),plain(heard)
    return difflib.SequenceMatcher(None,a,b).ratio() if a and b else 0.


def already_english(source,english):
    """True when the translation is the source text itself (ignoring case, punctuation and spacing): the line was English
    already, so re-speaking it would only replace the real actor's English with a copy."""
    def plain(text):return re.sub(r'[\W_]+',' ',str(text).casefold()).strip()
    return bool(plain(source)) and plain(source)==plain(english)


def translation_retry_budget(seconds):
    """Word budget for translating a line again after a truncated or empty answer: its own duration at a brisk pace, with a
    little room, never fewer than 4 words."""
    return max(4,min(60,int(float(seconds)*CONCISE_WORDS_PER_SECOND)+2))


def concise_budget(english,window):
    """Words an English line may use to fit `window` seconds (about 2.6 words/s natural pace plus the tempo fit), or None when the
    line already fits that budget or the slot is unusable."""
    count=len(re.findall(r"[\w']+",str(english)))
    budget=int(window*CONCISE_WORDS_PER_SECOND)
    return budget if window>0 and 1<=budget<count else None


def choose_take(path,receipt,window,*,render,fit,check=None,seeds=TAKE_RETRY_SEEDS,recoverable=lambda error:False,concise=None):
    """(path,receipt) is a take already passed through fit_rendered_take. It is accepted when it fits `window` and, if a
    `check(path,receipt) -> {'critical': bool, 'error': float}` (readback) is given, reads back with no critical change and
    an ordinary word error <= READBACK_ORDINARY_ERROR. Otherwise the same words are rendered on the next seed
    (render(seed) -> raw (path,receipt); fit(path,receipt) -> (path,receipt)) until one is accepted. When every attempt is too
    long for the slot and `concise() -> str | None` offers a shorter English wording of the same line, that wording is
    rendered the same way (render(seed,text)); its receipt records the change in receipt['concise']. When nothing is
    accepted the best attempt is kept: no critical readback change and not unintelligible, then fitting the slot, then
    clear speech, then a natural pace (tempo fit <= PREFERRED_FIT_FACTOR), then the shorter one. A retry failing with an error `recoverable` accepts leaves the attempts so far; cancellation, resource
    waits and every other error propagate. A changed result records every attempt in receipt['seed_retry']."""
    if window<=0:return path,receipt
    def judge(seed,got,text=None):
        seconds=float(got[1].get('seconds') or 0)
        verdict=check(*got) if check is not None else {'critical':False,'error':0.}
        entry={'seed':seed,'seconds':round(seconds,3),'overrun':seconds>window+1e-6,
               'critical':bool(verdict.get('critical')),'error':round(float(verdict.get('error') or 0.),3)}
        entry['fast']=float((got[1].get('fit') or {}).get('factor') or 1.)>PREFERRED_FIT_FACTOR
        if text is not None:entry['text']=text
        entry['accepted']=(not entry['overrun'] and not entry['critical'] and entry['error']<=READBACK_ORDINARY_ERROR
                           and not entry['fast'])
        return entry
    attempts=[(judge(42,(path,receipt)),(path,receipt),None)]
    def attempt(seed,text=None):
        try:got=fit(*(render(seed) if text is None else render(seed,text)))
        except Exception as error:
            if not recoverable(error):raise
            attempts.append(({'seed':seed,'error':type(error).__name__,'accepted':False,'unusable':True,
                              **({'text':text} if text is not None else {})},None,text))
            return False
        attempts.append((judge(seed,got,text),got,text))
        return attempts[-1][0]['accepted']
    for seed in seeds:
        if attempts[-1][0]['accepted']:break
        attempt(seed)
    usable=[item for item in attempts if item[1] is not None]
    if concise is not None and not any(item[0]['accepted'] for item in usable) and usable and all(item[0]['overrun'] for item in usable):
        shorter=concise()
        if isinstance(shorter,str) and shorter.strip():
            for seed in CONCISE_SEEDS:
                if attempt(seed,shorter.strip()):break
            usable=[item for item in attempts if item[1] is not None]
    accepted=[item for item in usable if item[0]['accepted']]
    best=accepted[0] if accepted else min(usable,key=lambda item:(item[0]['critical'] or item[0]['error']>=UNINTELLIGIBLE_ERROR,
        item[0]['overrun'],item[0]['error']>READBACK_ORDINARY_ERROR,item[0]['fast'],item[0]['seconds']))
    if len(attempts)==1:return best[1]
    chosen=dict(best[1][1],seed_retry={'window':round(window,3),'chosen_seed':best[0]['seed'],'accepted':best[0]['accepted'],
                                     'attempts':[item[0] for item in attempts]})
    if best[2] is not None:chosen['concise']={'text':best[2],'requested':receipt.get('requested_text')}
    return best[1][0],chosen


def adopt_spoken_text(cue,receipt):
    """A take of a shortened wording becomes the cue's English (so the readback and the ledger compare the words that were
    spoken), as adopt_fitted_text does for the Vox duration fit. Reused takes carry the same record."""
    concise=receipt.get('concise')
    if isinstance(concise,dict) and concise.get('text') and concise['text']!=cue.get('english'):
        cue.setdefault('english_requested',cue.get('english'))
        cue['english']=concise['text'];cue['english_basis']='concise-retranslation'


# protect_dialogue_mix fades edits over 10 ms outside each protected span.
PROTECTION_TRANSITION_SECONDS=.01
# Same guard schedule_lines keeps between generated lines.
PROTECTION_SCHEDULE_GAP_SECONDS=.04

# beta.5 ASR regression fix (actor_beta only; every legacy path is unchanged).
BETA5_ASR_FIX_VERSION='beta5-asr-gain-fix-v4'
# The two v3 helper modules are bound to this file: config.engine_fingerprint hashes actor_pipeline.py but not
# them, so their reviewed sha256 values live here and a drifted helper fails closed (caption_targets only).
OWNERSHIP_MODULE_SHA256={'beta_dialogue_ownership.py':'daaeba3162777300e2a366e0ce8a48b76a16cf4efcbd2028086c40cc1814ae93','beta_line_ledger.py':'c89081ade561b743ff61079ee3d2c1f08aa755b1c49eff4ad1a3c0050114c2a0'}
# A window whose line-ledger commit lost a race re-renders from a fresh snapshot at most this often.
LINE_COMMIT_ATTEMPTS=3
# v3 published == heard: before commit, every new or reused take must be audible in the RENDERED mix inside this
# window's render range. Measured on (rendered mix - the same mix without generated speech): per 20 ms block of the
# take's in-window placement, the take's own component (projection onto the expected placed take) must keep at least
# -6 dB of the expected level and exceed -60 dBFS, in >= 90% of the blocks where the take itself is above -60 dBFS.
AUDIBLE_BLOCK_SECONDS=.02
AUDIBLE_FLOOR_DBFS=-60.
AUDIBLE_MAX_LOSS_DB=6.
AUDIBLE_MIN_BLOCK_FRACTION=.9
# ... AND it must be heard over the rest of the rendered mix (W2 P3-3: not attenuated is not the same as audible). In
# the take's voiced blocks (within 20 dB of its loudest block) the take component must be no more than 6 dB below
# everything else in the output at that moment (mix minus the take), in >= 80% of them.
AUDIBLE_VOICED_RANGE_DB=20.
AUDIBLE_MIN_TAKE_TO_REST_DB=-6.
AUDIBLE_MIN_VOICED_FRACTION=.8
AUDIBLE_POLICY='rendered-take-projection-v2'
# A take that starts a few milliseconds before the window ends has a sliver of itself in this window; judged on that sliver it was
# boosted by up to 24 dB and carried into the next window at that gain (chunk 205 of the 2026-10-20 run, +18 LU). The window that
# holds the rest of the take judges it.
AUDIBLE_MIN_IN_WINDOW_SECONDS=.5
# Owner decision 2026-10-19: a take masked by loud music or effects, or so quiet that it sits below the audibility floor (a
# whisper the original mix barely carries), is raised, not dropped. Each re-render lifts it by this step until it is heard over
# the rest of the mix, never beyond the cap (a take attenuated by the mix itself is still dropped).
AUDIBLE_BOOST_STEP_DB=6.
AUDIBLE_BOOST_MAX_DB=12.
AUDIBLE_BOOST_REASONS=frozenset({'take masked by the rest of the rendered mix','take is silent inside the render range'})
# The no-speech bed for that measurement is the SAME protect_dialogue_mix function, bound here at import, so the
# window's own mix stays one call of the module-level name (tests wrap it to inspect exactly that call).
_bed_mix=protect_dialogue_mix
# plan_dialogue caption-edge split (caption_targets only): a real pause of at
# least this length that contains a caption start or end starts a new cue.
CAPTION_EDGE_PAUSE_SECONDS=.25
LINE_REFERENCE_VERSION='line-own-reference-v1'
BANK_REFERENCE_MIN_SECONDS=3.0
VOICE_POLICY_VERSION='own-line-first-bank-short-v1'


def line_voice_config(config):
    """Use the existing caption_targets switch for this lane without changing owner configuration.

    Callers still use the original switch for cross-window ownership and line ledgers.
    This view selects per-cue caption wording and the stable per-cue admission gate.
    """
    if _beta.beta_settings(config) is None:return config
    view=_beta.ConfigView(config)
    view.data=view
    view['actor_beta']={**view['actor_beta'],'caption_targets':False}
    return view
# relay.first_pass.processing_stereo (selected by source layout; the label is
# relay.first_pass_key.TRANSFORMS[layout]) gives a centre-panned source this
# coefficient in mean(L,R): mono m -> [m/sqrt2]*2; stereo phantom centre
# L=R=c -> identity; 5.1/5.1(side) FC -> 2**-.5*C/(1+2**.5) in both L and R
# (ITU, no LFE). The candidate tests re-derive every value through processing_stereo.
FIRST_PASS_CENTRE_COEFFICIENT={'mono':2**-.5,'stereo':1.0,'5.1':2**-.5/(1+2**.5),'5.1(side)':2**-.5/(1+2**.5)}


def first_pass_centre_gain(source_layout):
    """Gain returning centre-channel dialogue in mean(L,R) of the first-pass
    processing stereo to unity. An unknown source layout fails closed."""
    coefficient=FIRST_PASS_CENTRE_COEFFICIENT.get(source_layout)
    if coefficient is None:
        raise ValueError('No centre-gain derivation for first-pass source layout %r'%(source_layout,))
    return 1/coefficient


def ownership_modules_verified():
    """caption_targets: the two v3 helper modules must be the reviewed bytes bound in OWNERSHIP_MODULE_SHA256."""
    for module in (_own,_lines):
        name=Path(module.__file__).name
        if file_hash(Path(module.__file__))!=OWNERSHIP_MODULE_SHA256.get(name):
            raise ValueError('%s does not match the version bound to actor_pipeline.py'%name)
    return True


# Channel-aware ASR input (first pass only). The mid (L+R)/2 is used only when
# the window's channels are in phase and the mid keeps the stronger channel's
# energy; otherwise the strongest single channel (the beta.5 selection, which
# never cancels phase-opposed dialogue). Thresholds are on the 48 kHz raw vocals.
ASR_INPUT_POLICY='channel-aware-centre-gain-v1'
ASR_MID_MIN_CORRELATION=.4       # window L/R correlation below this: single channel
ASR_MID_MAX_DEFICIT_DB=-3.       # window mid energy this far below the stronger channel: single channel
ASR_BLOCK_SECONDS=.5             # local check: any active block whose mid loses
ASR_BLOCK_CANCEL_DB=-9.          # more than 9 dB against its stronger channel
ASR_BLOCK_ACTIVE_RANGE_DB=30.    # active = within 30 dB of the loudest block
ASR_BLOCK_FLOOR_DBFS=-60.        # and above -60 dBFS (stronger channel, before gain)
# Limiter at the 16 kHz analysis signal: identity up to the knee, then a tanh
# knee bounded by the ceiling (one PCM_16 step below full scale).
ASR_LIMITER_KNEE=.9
ASR_LIMITER_CEILING=32767/32768
# A mid that cancels completely is reported at this finite floor (strict JSON receipts).
ASR_DEFICIT_FLOOR_DB=-300.
# Level of a one-sided source in the mid: (s+0)/2 keeps a quarter of the energy, 20*log10(1/2) = -6.02 dB. Any
# block whose mid is lower than that has negatively correlated channels (cancellation); ASR_BLOCK_CANCEL_DB only
# fires there. One-sided lines inside an in-phase window stay on the mid at -6.02 dB (+4.65 dB after the centre
# gain) on purpose: switching the whole window to one channel would drop a line on the other side completely.
ASR_ONE_SIDED_MID_DB=-6.020599913279624


def _energy_db(value):
    import math
    return 10*math.log10(value) if value>0 else None


def _deficit_db(mid,strong):
    import math
    return max(ASR_DEFICIT_FLOOR_DB,10*math.log10(mid/strong)) if mid>0 else ASR_DEFICIT_FLOOR_DB


def asr_channel_metrics(data,rate):
    """L/R correlation, per-channel and mid energy, and the worst active block."""
    import numpy as np
    left,right=data[:,0],data[:,1]
    el,er=float(np.mean(left*left)) if len(left) else 0.,float(np.mean(right*right)) if len(right) else 0.
    mid=(left+right)*.5
    em=float(np.mean(mid*mid)) if len(mid) else 0.
    strong=max(el,er)
    correlation=float(np.sum(left*right)/np.sqrt(np.sum(left*left)*np.sum(right*right))) if el>0 and er>0 else 0.
    step=max(1,int(round(ASR_BLOCK_SECONDS*rate)));blocks=[]
    for start in range(0,len(left),step):
        l,r=left[start:start+step],right[start:start+step]
        a,b=float(np.mean(l*l)),float(np.mean(r*r));m=float(np.mean(((l+r)*.5)**2))
        blocks.append((start/rate,a,b,m))
    loudest=max([max(a,b) for _,a,b,_ in blocks],default=0.)
    active=[(t,a,b,m) for t,a,b,m in blocks if loudest>0 and max(a,b)>0 and
            10*np.log10(max(a,b)/loudest)>=-ASR_BLOCK_ACTIVE_RANGE_DB and 10*np.log10(max(a,b))>=ASR_BLOCK_FLOOR_DBFS]
    worst=None
    for t,a,b,m in active:
        deficit=_deficit_db(m,max(a,b))
        if worst is None or deficit<worst['mid_deficit_db']:
            worst={'offset_seconds':t,'mid_deficit_db':float(deficit),'stronger_dbfs':float(10*np.log10(max(a,b)))}
    return {'correlation':correlation,'left_db':_energy_db(el),'right_db':_energy_db(er),'mid_db':_energy_db(em),
            'mid_deficit_db':_deficit_db(em,strong) if strong>0 else None,
            'stronger_channel':0 if el>=er else 1,'silent':strong<=0,'blocks':len(blocks),'active_blocks':len(active),
            'worst_active_block':worst}


def asr_channel_choice(metrics):
    """('mid'|'channel', reason) from asr_channel_metrics."""
    if metrics['silent']:return 'mid','silent window (nothing to cancel)'
    if metrics['correlation']<ASR_MID_MIN_CORRELATION:return 'channel','L/R correlation %.3f below %.2f'%(metrics['correlation'],ASR_MID_MIN_CORRELATION)
    if metrics['mid_deficit_db']<ASR_MID_MAX_DEFICIT_DB:return 'channel','mid %.2f dB below the stronger channel'%metrics['mid_deficit_db']
    worst=metrics['worst_active_block']
    if worst and worst['mid_deficit_db']<ASR_BLOCK_CANCEL_DB:
        return 'channel','active block at +%.1f s loses %.2f dB in the mid'%(worst['offset_seconds'],worst['mid_deficit_db'])
    return 'mid','in phase (correlation %.3f, mid %.2f dB)'%(metrics['correlation'],metrics['mid_deficit_db'])


def asr_soft_limit(signal):
    """Identity up to ASR_LIMITER_KNEE; above it a tanh knee bounded by ASR_LIMITER_CEILING."""
    import numpy as np
    out=np.array(signal,dtype=np.float64,copy=True)
    magnitude=np.abs(out);over=magnitude>ASR_LIMITER_KNEE
    room=ASR_LIMITER_CEILING-ASR_LIMITER_KNEE
    out[over]=np.sign(out[over])*(ASR_LIMITER_KNEE+room*np.tanh((magnitude[over]-ASR_LIMITER_KNEE)/room))
    return out,int(np.count_nonzero(over)),float(magnitude.max()) if len(magnitude) else 0.


ASR_LOCAL_GAIN_VERSION='asr-local-gain-v1'


def asr_local_gain(signal,rate=16000,*,target_db=-23.0,gate_db=-58.0,max_gain_db=30.0):
    """Owner design: quiet dialogue is amplified so the recognizer can hear it (analysis input only; the mix, references and
    dub level never use this). A slow, smoothed gain per quarter-second raises each speech-bearing stretch toward target_db
    RMS by at most max_gain_db and never lowers anything; stretches below gate_db (silence, noise floor) are not boosted.
    Returns (float64 signal of the same length, receipt)."""
    import numpy as np
    x=np.asarray(signal,dtype=np.float64)
    hop=int(rate)//4
    empty={'version':ASR_LOCAL_GAIN_VERSION,'applied':False,'target_db':target_db,'gate_db':gate_db,'max_gain_db':max_gain_db,
           'mean_gain_db':0.0,'max_applied_db':0.0,'boosted_seconds':0.0}
    if x.ndim!=1 or len(x)<2*hop or not np.isfinite(x).all():return x,empty
    count=len(x)//hop+1
    level=np.full(count,-120.0)
    for index in range(count):
        window=x[index*hop:(index+2)*hop]
        if len(window):level[index]=20*np.log10(float(np.sqrt(np.mean(window**2)))+1e-9)
    active=level>gate_db
    if not active.any():return x,empty
    positions=np.arange(count)
    wanted=np.clip(target_db-level,0.0,max_gain_db)
    wanted=np.interp(positions,positions[active],wanted[active])      # inactive frames inherit the nearest speech frame's gain
    wanted=np.convolve(np.pad(wanted,2,mode='edge'),np.ones(5)/5,mode='valid')[:count]
    per_sample=10.0**(np.interp(np.arange(len(x))/hop,positions,wanted)/20.0)
    # Silence between speech must stay silence: fade the inherited gain out below the gate (no pumped noise floor).
    quiet=np.interp(np.arange(len(x))/hop,positions,(level<=gate_db).astype(float))
    per_sample=1.0+(per_sample-1.0)*(1.0-quiet)
    boosted=wanted>0.5
    return x*per_sample,{'version':ASR_LOCAL_GAIN_VERSION,'applied':True,'target_db':target_db,'gate_db':gate_db,
        'max_gain_db':max_gain_db,'mean_gain_db':float(np.mean(wanted[active])),'max_applied_db':float(np.max(wanted)),
        'boosted_seconds':float(np.count_nonzero(boosted&active))*hop/rate}


def asr_local_gain_enabled(config):
    """actor_beta.asr_local_gain (default on): amplify quiet dialogue for the recognizer."""
    return (_beta.beta_settings(config) or {}).get('asr_local_gain',True) is not False


def first_pass_recognition(raw_vocals,source_layout,transform,rate=48000,*,local_gain=False):
    """16 kHz ASR/speaker analysis input from the first-pass vocals stem.

    Channel-aware: the gain-corrected mid for in-phase windows, else the
    strongest single channel (asr_channel_choice). Both carry the centre
    compensation of the processing downmix (the centre coefficient is the same
    in L, R and the mid), then the soft limiter at 16 kHz. Only analysis
    changes: mix, references and pan still use raw_vocals.
    """
    import math
    import numpy as np
    from scipy.signal import resample_poly
    data=np.asarray(raw_vocals,dtype=np.float64)
    if data.ndim!=2 or data.shape[1]!=2:raise ValueError('First-pass vocals must be stereo')
    # Analysis only: a non-finite sample becomes silence (counted) so metrics, ASR input and receipt stay finite.
    finite=np.isfinite(data);nonfinite=int(data.size-np.count_nonzero(finite))
    if nonfinite:data=np.where(finite,data,0.)
    gain=first_pass_centre_gain(source_layout)
    metrics=asr_channel_metrics(data,rate)
    choice,reason=asr_channel_choice(metrics)
    if choice=='mid':
        recognition=resample_poly((data[:,0]+data[:,1])*(.5*gain),1,3);mix='(L+R)/2 of first-pass raw_vocals'
    else:
        channel=metrics['stronger_channel']
        recognition=resample_poly(data[:,channel]*gain,1,3);mix='strongest single channel %s of first-pass raw_vocals'%'LR'[channel]
    gain_receipt=None
    if local_gain:recognition,gain_receipt=asr_local_gain(recognition,16000)
    recognition,limited,peak=asr_soft_limit(recognition)
    return recognition.astype('float32'),{'version':BETA5_ASR_FIX_VERSION,'policy':ASR_INPUT_POLICY,'choice':choice,'local_gain':gain_receipt,
        'reason':reason,'mix':mix,'metrics':metrics,'thresholds':{'min_correlation':ASR_MID_MIN_CORRELATION,
        'max_mid_deficit_db':ASR_MID_MAX_DEFICIT_DB,'block_seconds':ASR_BLOCK_SECONDS,'block_cancel_db':ASR_BLOCK_CANCEL_DB,
        'block_active_range_db':ASR_BLOCK_ACTIVE_RANGE_DB,'block_floor_dbfs':ASR_BLOCK_FLOOR_DBFS},
        'source_layout':source_layout,'transform':transform,'centre_gain':gain,'centre_gain_db':20*math.log10(gain),
        'limiter':'identity to %.2f, tanh knee to %.6f, at 16 kHz'%(ASR_LIMITER_KNEE,ASR_LIMITER_CEILING),'limited_samples':limited,
        'peak_before_limiter':peak,'clipped_samples':0,'nonfinite_samples':nonfinite,'samples':int(len(recognition)),
        'one_sided_mid_db':ASR_ONE_SIDED_MID_DB}


def keep_caption_issue(caption,render_start,render_end,*,speech_regions=None,by_speech_start=False):
    """Legacy: a caption issue counts in every window its caption overlaps.
    by_speech_start (caption_targets): only the window holding the caption's
    speech start, the same speech-start rule that owns every cue
    (beta_dialogue_ownership.keep_caption_issue)."""
    if by_speech_start:return _own.keep_caption_issue(caption,speech_regions,render_start,render_end)
    return caption['end']>render_start and caption['start']<render_end


def classify_original_audio(all_cues,speech_regions,issues,*,origin,end,render_start,render_end,canonical=()):
    """Plan kept-original exceptions over every source cue and every issue.

    ``issues`` is the complete cumulative issue list and is extended in place
    with cue-scoped blockers for a keep that this window cannot own safely:
    its protected span plus transition must lie inside the render range (the
    only output this window publishes), and it must not coincide with a
    canonical generated take already committed by another window. Such a cue
    is then not kept, so its original issues remain blockers.
    """
    while True:
        policy=plan_original_audio(all_cues,speech_regions,issues,origin=origin,end=end)
        blockers=[]
        for row in policy['protected_intervals']:
            begin,stop=row['start']-PROTECTION_TRANSITION_SECONDS,row['end']+PROTECTION_TRANSITION_SECONDS
            if begin<render_start or stop>render_end:
                blockers.append({'kind':'kept_original_crosses_render_range','cue':row['cue'],
                                 'start':row['start'],'end':row['end'],
                                 'render_start':render_start,'render_end':render_end})
                continue
            for item in canonical:
                left=min(item['start'],item['replacement_start'],item['source_start'])
                right=max(item['end'],item['replacement_end'],item['source_end'])
                if min(stop,right)>max(begin,left):
                    blockers.append({'kind':'kept_original_conflicts_with_canonical_take','cue':row['cue'],
                                     'canonical_id':item['id'],'start':row['start'],'end':row['end']})
                    break
        if not blockers:return policy
        issues.extend(blockers)


def ownership_accounted(report,all_cues):
    """v2 (caption_targets): source cues in the render range that another window owns.

    The policy validator requires every render-range source cue to be
    synthesized or explicitly kept. With ownership, a non-owned cue is instead
    accounted by a boundary item of this window (protected original or the
    owner's published take, decided per span); it is presented to the
    validator as accounted, never synthesized. Without ownership: [].
    """
    ownership=(report or {}).get('beta_dialogue_ownership')
    if ownership is None:return []
    render_start,render_end=report['render_start'],report['render_end']
    listed={ident for item in ownership['items'] for ident in item['cue_ids']}
    accounted=[]
    for cue in all_cues or ():
        row=ownership['cues'].get(cue['id'])
        if row is None:raise ValueError('A source cue has no dialogue-ownership decision')
        if row['owned'] or not (cue['end']>render_start and cue['start']<render_end):continue
        if cue['id'] not in listed:raise ValueError('A non-owned render-range cue is not accounted by a boundary item')
        accounted.append({'id':cue['id'],'start':cue['start'],'end':cue['end']})
    return accounted


def boundary_render_plan(report,ownership,cues,all_cues,canonical,ledger,source_identity,claim=None):
    """v3 (caption_targets): per-span boundary decisions at render time.

    Decided against the canonical takes that exist now (a reused plan may
    predate the owner's publication). Only this window's own canonical take
    of a line is reused; an owned unit whose source is the same line as any
    other canonical take is not synthesized again (a second claim, e.g. ASR
    start jitter at a boundary), and its span is decided like a boundary item
    (covered by that take, else protected original). Every other owned unit
    that needs a new take is claimed in the atomic line ledger BEFORE any
    synthesis (``claim``: beta_line_ledger.claim_lines); a unit whose line is
    held or published by another claimant is not synthesized either and is
    decided the same way. Protected fallback spans join the runtime walls, and
    a synthesis cue overlapping one keeps original (cascading like the beta
    fallback). Returns (cues, spans, line records, claim results by cue id).
    """
    render_start,render_end=report['render_start'],report['render_end']
    window=[render_start,render_end]
    lines={c['id']:_own.line_record(c,render_start=render_start,render_end=render_end,source_identity=source_identity)
           for c in cues}
    line_ids={ident:row['line_id'] for ident,row in lines.items()}
    def prior(cue):
        return _own.own_prior(cue,canonical,window=window,line_ids=line_ids,match=ledger.match)
    conflicts=_own.canonical_conflicts(cues,canonical,prior=prior)
    items=list(ownership['items'])
    def item(cue,kind,**extra):
        caption=_own.unit_caption(cue)
        spans=[[float(a),float(b)] for a,b in cue.get('authorized_audio_spans') or [[cue['start'],cue['end']]]]
        return {'item_id':line_ids[cue['id']],'kind':kind,'line_start':float(cue['start']),
                'owner':'this_window_by_its_own_asr','caption':None if not caption else
                {'start':float(caption['start']),'end':float(caption['end'])},
                'caption_key':lines[cue['id']]['line_caption_key'],
                'cue_ids':[cue['id']]+[c for c in cue.get('source_cue_ids') or () if c!=cue['id']],
                'cue_spans':spans,'speech_pieces':_own.clip(spans,render_start,render_end),
                'speech_inside_owned_cues':[],'speech_basis':'owned unit source span',**extra}
    for cue,ids in conflicts:
        items.append(item(cue,'owned_cue_canonical_overlap',conflicting_canonical_ids=ids))
    dropped={cue['id'] for cue,_ in conflicts}
    claimed={}
    if claim is not None:
        wanted=[c for c in cues if c['id'] not in dropped and prior(c) is None]
        results=claim([{'line_id':lines[c['id']]['line_id'],'source_span':list(lines[c['id']]['line_source_span']),
                        'take_id':c['id'],'line_window':lines[c['id']]['line_window'],
                        'line_caption_key':lines[c['id']]['line_caption_key']} for c in wanted]) if wanted else {}
        for cue in wanted:
            result=results[line_ids[cue['id']]]
            claimed[cue['id']]=result
            if result['status'] in ('held','published'):
                items.append(item(cue,'owned_unit_claimed_elsewhere',claim_status=result['status'],
                                  blocking_claim=result['blocking']))
                dropped.add(cue['id'])
        ownership['line_claims']=[{'cue':ident,'line_id':r['line_id'],'status':r['status'],'token':r['token'],
                                   'blocking':r['blocking'],'artifacts':sorted(r.get('artifacts') or ())}
                                  for ident,r in claimed.items()]
    cues=[c for c in cues if c['id'] not in dropped]
    decisions,walls=_own.decide(items,canonical,render_start=render_start,render_end=render_end,
        canonical_intervals=_ctf.canonical_intervals,take_span=take_span,evaluated_at='render')
    ownership['decisions']=decisions
    ownership['render_canonical_conflicts']=[{'cue':cue['id'],'canonical_ids':ids} for cue,ids in conflicts]
    runtime=report.setdefault('beta_runtime_fallback',{})
    spans=list(walls)
    if walls:
        prepared=report.get('beta_original_fallback') or {}
        while True:
            spans=merge_spans(walls,_beta.fallback_spans({**prepared,**runtime},all_cues))
            hit=[c for c in cues if c['id'] not in runtime and
                 any(min(float(c['end']),b)>max(float(c['start']),a) for a,b in spans)]
            if not hit:break
            for cue in hit:runtime.setdefault(cue['id'],[]).append('overlaps_boundary_original_fallback')
        report['beta_fallback_intervals']=[list(s) for s in spans]
        cues=[c for c in cues if c['id'] not in runtime]
    ownership['protected_intervals']=[list(s) for s in walls]
    return cues,spans,{ident:row for ident,row in lines.items() if ident not in dropped},claimed


# Claim results that allow this window to synthesize a line (beta_line_ledger.claim_lines).
SYNTHESIS_GRANTS=('acquired','reacquired','taken_over')


def chunk_fence(store,job_id,revision=1):
    """v4: the coordinator's chunk claim this worker runs under (job id, chunk index, claim token, job revision), or
    None. coordinator._ClaimedStore holds it. Every line-ledger write, receipt write and speaker-bank write re-checks
    it inside its own transaction (beta_line_ledger.Superseded), and every claim stores it."""
    key,token=getattr(store,'_key',None),getattr(store,'_token',None)
    if not isinstance(key,tuple) or len(key)!=2 or isinstance(token,bool) or not isinstance(token,int):return None
    if key[0]!=job_id:return None
    return _lines.fence_of(key[0],key[1],token,revision)


class FencedStore:
    """v4 (W2 P2-5): the store a fenced window's speaker bank writes through. put_speaker runs Store.put_speaker's
    exact statement inside one IMMEDIATE transaction that first re-checks the chunk fence, so a superseded worker
    cannot write actor identity rows. Every other attribute is the wrapped store, unchanged."""
    def __init__(self,store,fence):
        self._inner,self._fence=store,fence

    def __getattr__(self,name):
        return getattr(self._inner,name)

    def put_speaker(self,job_id,ident,payload):
        with self._inner.connect() as db:
            _lines._begin(db,self._fence)
            db.execute("INSERT INTO speakers VALUES(?,?,?) ON CONFLICT(job_id,id) "
                       "DO UPDATE SET payload=excluded.payload",(job_id,ident,json.dumps(payload)))


def fenced_json(store,fence,path,payload):
    """v4 (W2 P2-4): a receipt/plan JSON write under the chunk fence (beta_line_ledger.fenced_write)."""
    return _lines.fenced_write(store,fence,lambda:atomic_json(path,payload))


class LineClaimLost(Exception):
    """v3: this run's line claim was released or taken over while it synthesized; the take is not published."""


class InaudibleTakes(Exception):
    """v3: a take that would be published is not audible in the rendered mix; the window re-renders with it kept
    original (it is never committed)."""
    def __init__(self,cue_ids,rows):
        super().__init__('takes not audible in the rendered mix: %s'%sorted(cue_ids))
        self.cue_ids,self.rows=list(cue_ids),rows


def _placed(samples,left,pan,gain,a,b):
    """Frames [a, b) (window-relative) of a mono take placed at ``left`` with pan/gain, zero elsewhere."""
    import numpy as np
    pan=np.asarray(pan,dtype=np.float64)
    out=np.zeros((max(0,b-a),len(pan)))
    lo,hi=max(a,left),min(b,left+len(samples))
    if hi>lo:out[lo-a:hi-a]=np.asarray(samples[lo-left:hi-left],dtype=np.float64)[:,None]*pan[None,:]*float(gain)
    return out


def rendered_audibility(mixed,bed,origin,rate,render_start,render_end,takes):
    """v3: measure each take in the RENDERED output. ``bed`` is the same protect_dialogue_mix call with no generated
    speech, so mixed - bed is exactly the generated speech the viewer hears (after protection masks and headroom
    limiting). ``takes``: [{'take_id','kind','start','end','samples','left','pan','gain','gating'}] with ``left`` the
    window-relative frame where the mix placed it. Per 20 ms block of the take's in-render-range placement the take's
    own component is projected out of mixed - bed; see AUDIBLE_* for the rule. Strict JSON floats only."""
    import math
    import numpy as np
    floor=10**(AUDIBLE_FLOOR_DBFS/20);keep=10**(-AUDIBLE_MAX_LOSS_DB/20)
    step=max(1,int(round(AUDIBLE_BLOCK_SECONDS*rate)))
    def db(x):return round(20*math.log10(x),3) if x>0 else None
    rows={}
    for take in takes:
        s,e=float(take['start']),float(take['end'])
        lo,hi=max(s,float(render_start)),min(e,float(render_end))
        row={'take_id':take['take_id'],'kind':take['kind'],'start':s,'end':e,'gating':bool(take['gating']),
             'in_window':[lo,hi] if hi>lo else None,'policy':AUDIBLE_POLICY}
        a,b=max(0,int(round((lo-origin)*rate))),min(len(mixed),int(round((hi-origin)*rate)))
        if take['kind']=='generated' and 0<hi-lo<AUDIBLE_MIN_IN_WINDOW_SECONDS and e-s>=2*(hi-lo):
            rows[take['take_id']]={**row,'audible':True,'reason':'only a sliver of the take is inside this window; the window holding the rest judges it',
                                   'active_blocks':0,'audible_blocks':0,'sliver_seconds':round(hi-lo,3)}
            continue
        if hi<=lo or b<=a:
            # A carried take may touch this window only through its removal envelope: nothing to hear here.
            rows[take['take_id']]={**row,'audible':None if take['kind']=='carried' else False,
                                   'reason':'no part of the take is inside the render range',
                                   'active_blocks':0,'audible_blocks':0}
            continue
        expected=_placed(take['samples'],int(take['left']),take['pan'],take['gain'],a,b)
        heard=np.asarray(mixed[a:b],dtype=np.float64)-np.asarray(bed[a:b],dtype=np.float64)
        output=np.asarray(mixed[a:b],dtype=np.float64)
        blocks=[]
        for k in range(0,b-a,step):
            x,y,z=expected[k:k+step],heard[k:k+step],output[k:k+step]
            ee=float(np.sum(x*x))
            exp_rms=math.sqrt(ee/len(x))
            if exp_rms<floor:continue
            coefficient=float(np.sum(x*y))/ee
            rest=z-max(0.,coefficient)*x                       # everything else the viewer hears in this block
            blocks.append((exp_rms,coefficient,ee,math.sqrt(float(np.sum(rest*rest))/len(x))))
        active=len(blocks);good=0;proj_energy=exp_energy=0.
        loudest=max([e for e,_,_,_ in blocks],default=0.)
        voiced=heard_over=0;margin=10**(AUDIBLE_MIN_TAKE_TO_REST_DB/20);ratios=[]
        for exp_rms,coefficient,ee,rest_rms in blocks:
            take_rms=max(0.,coefficient)*exp_rms
            proj_energy+=max(0.,coefficient)**2*ee;exp_energy+=ee
            if coefficient>=keep and take_rms>=floor:good+=1
            if exp_rms>=loudest*10**(-AUDIBLE_VOICED_RANGE_DB/20):
                voiced+=1
                ratios.append(20*math.log10(take_rms/rest_rms) if take_rms>0 and rest_rms>0 else (99. if take_rms>0 else -99.))
                if take_rms>0 and take_rms>=margin*rest_rms:heard_over+=1
        fraction=good/active if active else 0.
        voiced_fraction=heard_over/voiced if voiced else 0.
        total_rendered=math.sqrt(float(np.sum(heard*heard))/(b-a))
        total_expected=math.sqrt(float(np.sum(expected*expected))/(b-a))
        kept=active>0 and fraction>=AUDIBLE_MIN_BLOCK_FRACTION
        over=voiced>0 and voiced_fraction>=AUDIBLE_MIN_VOICED_FRACTION
        audible=kept and over
        ratios.sort()
        rows[take['take_id']]={**row,'audible':audible,'not_attenuated':kept,'heard_over_mix':over,
                               'active_blocks':active,'audible_blocks':good,
                               'audible_fraction':round(fraction,4),'voiced_blocks':voiced,'voiced_heard_over_mix':heard_over,
                               'voiced_heard_fraction':round(voiced_fraction,4),
                               'take_to_rest_db_median':round(ratios[len(ratios)//2],3) if ratios else None,
                               'rendered_dbfs':db(total_rendered),
                               'expected_dbfs':db(total_expected),
                               'take_component_loss_db':db(math.sqrt(proj_energy/exp_energy)) if exp_energy>0 and proj_energy>0 else None,
                               **({} if audible else {'reason':'take is silent inside the render range' if not active
                                                       else 'take attenuated in the rendered mix' if not kept
                                                       else 'take masked by the rest of the rendered mix'})}
    return rows


class LineCommitRace(Exception):
    """v3: another window committed first (beta_line_ledger.races); the window re-renders from a fresh snapshot."""
    def __init__(self,races):
        super().__init__('line ledger commit lost a race: %s'%sorted({r['kind'] for r in races}))
        self.races=races


def line_ledger_state(store,job_id,revision,report,origin,end,canonical):
    """v3 (caption_targets): what other windows had committed when this render started. A new take may end after
    render_end only up to the first other committed window there (it rendered without the take)."""
    render_start,render_end=report['render_start'],report['render_end']
    key=_lines.window_key(render_start,render_end)
    windows=_lines.windows(store,job_id,revision,origin-1.,end+1.)
    others=[w for w in windows if w['window_key']!=key]
    state={'version':_lines.VERSION,'window_key':key,'placement_floor':float(render_start),
           'new_take_end_limit':_own.overrun_limit(windows,window_key=key,render_end=render_end,end=end),
           'snapshot_canonical_ids':sorted(i['id'] for i in canonical),
           'committed_other_windows':[{'window_key':w['window_key'],'render':[w['render_start'],w['render_end']]}
                                      for w in others]}
    report.setdefault('beta_line_ledger',{}).update(state)
    return {**state,'windows':windows,'audibility':{}}


def line_audibility_gate(report,cues,lines,placements,all_cues,line_state):
    """v3: before the mix, every scheduled take must be audible where it is placed: not before this window's render
    start and not ending inside a committed window that did not mix it. Any other placement keeps its cue original
    (cascading like the beta fallback). By construction (floor, end limit) this only guards reused takes."""
    render_start,render_end=report['render_start'],report['render_end']
    runtime=report.setdefault('beta_runtime_fallback',{});moved=False
    for cue,line,placement in zip(cues,lines,placements):
        take=line.get('canonical_id')
        limit=(_own.overrun_limit(line_state['windows'],window_key=line_state['window_key'],render_end=render_end,
                                  end=report['end'],take_id=take) if take else line_state['new_take_end_limit'])
        heard=_own.audibility(placement,render_start=render_start,render_end=render_end,limit=limit)
        line_state['audibility'][cue['id']]=heard
        if heard['problems']:
            runtime.setdefault(cue['id'],[]).extend('placement_not_audible:'+p for p in heard['problems']);moved=True
    if not moved:return cues,lines,placements,[]
    cues,lines,placements,spans=_beta._cascade(report,cues,lines,placements,all_cues)
    report['placements']=placements
    return cues,lines,placements,spans


def line_window_record(report,line_state,origin,end,lines,carry,pending):
    reused=sorted({line['canonical_id'] for line in lines if line.get('canonical_id')})
    return {'window_key':line_state['window_key'],'render_start':float(report['render_start']),
            'render_end':float(report['render_end']),'origin':float(origin),'end':float(end),
            'new_take_end_limit':line_state['new_take_end_limit'],
            'mixed_canonical_ids':sorted(set(reused)|{item['id'] for item in carry}),
            'new_take_ids':[cue['id'] for cue,*_ in pending],'attempt':(report.get('beta_line_ledger') or {}).get('attempt'),
            'owned_unit_spans':line_state.get('owned_unit_spans',[])}


def line_claims(pending,line_state,claimed):
    """The line claims this window publishes: one per new take, with the token it was granted before synthesis."""
    out=[]
    for cue,line,placement,_,_ in pending:
        if not line.get('line_id'):raise ValueError('A new owned take has no line record')
        grant=claimed.get(cue['id'])
        if grant is None or grant['status'] not in SYNTHESIS_GRANTS:raise ValueError('A new owned take has no held line claim')
        out.append({'line_id':line['line_id'],'take_id':cue['id'],'token':grant['token'],
                    'source_span':list(line['line_source_span']),
                    'line_window':line['line_window'],'line_caption_key':line.get('line_caption_key'),
                    'placement':[float(placement['start']),float(placement['end'])],
                    'audible':line_state['audibility'].get(cue['id']),
                    'rendered':line_state['rendered'].get(cue['id'])})
    return out


def boundary_claims(ownership):
    return [{'item_id':d['item_id'],'decision':d['decision'],'reason':d.get('reason'),'spans':d['protected']}
            for d in ownership.get('decisions') or () if d.get('protected')]


def annotate_line_audibility(report,line_state):
    """v3: each published placement carries where it is heard and the rendered-mix measurement that proved it.
    'published' means audible: placed at or after this window's render start, measured audible in this window's
    rendered output (rendered_audibility), and any part after render_end committed for a later window's carry with no
    committed window rendered without it. A row without that proof is demoted (published False) everywhere it is
    listed; the gate before commit makes that a fail-safe that should never fire."""
    rows,bad=[],[]
    for row in report.get('placements') or ():
        if not row.get('published'):continue
        heard=line_state['audibility'].get(row['cue_id']);rendered=line_state['rendered'].get(row['cue_id'])
        row['audible']=None if heard is None else {**heard,'rendered':rendered}
        if (heard is None or heard['problems'] or not heard['in_window'] or rendered is None
                or not rendered['audible']):bad.append(row['cue_id'])
        rows.append(row['cue_id'])
    if bad:
        for row in report.get('placements') or ():
            if row['cue_id'] in bad:row.update(published=False,fallback=['not_audible_in_render'])
        report['published_cue_ids']=[c for c in report.get('published_cue_ids') or () if c not in bad]
        report['published_utterances']=[u for u in report.get('published_utterances') or () if u['cue_id'] not in bad]
        mapping=report.get('published_mapping') or {}
        # The take was already mixed and committed, so the receipt cannot claim original audio for it: the mapping is
        # marked inconsistent and B's scorer (it requires status 'ok') refuses the whole receipt. Fail closed.
        if mapping.get('status')=='ok':
            mapping.update(published=len(report['published_cue_ids']),not_audible_demoted=bad,status='inconsistent',
                           reason='published take(s) without rendered audibility proof: %s'%sorted(bad))
    for row in report.get('published_utterances') or ():
        row['rendered_audibility']=line_state['rendered'].get(row['cue_id'])      # read by B's scorer (RESULTS.md)
    carried=[r for r in line_state['rendered'].values() if r['kind']=='carried']
    report['beta_line_ledger'].update(status='inconsistent' if bad else 'committed',published_checked=rows,
                                      published_not_audible=bad,audible_policy=AUDIBLE_POLICY,
                                      rendered_audibility=sorted(line_state['rendered'].values(),key=lambda r:r['start']),
                                      carried_not_audible=[r['take_id'] for r in carried if r['audible'] is False])


def record_no_dialogue_window(store,job_id,revision,report,origin,end,claimant=None,fence=None):
    """v3: commit the window record of a no-dialogue window (no speech, no cues, no canonical take in its range).
    False when a relevant canonical take was committed after this window read the ledger; it must render."""
    key=_lines.window_key(report['render_start'],report['render_end'])
    window={'window_key':key,'render_start':float(report['render_start']),'render_end':float(report['render_end']),
            'origin':float(origin),'end':float(end),'new_take_end_limit':float(end),'mixed_canonical_ids':[],
            'new_take_ids':[],'attempt':1,'owned_unit_spans':[],'mode':'no-dialogue'}
    problems,races=_lines.commit(store,job_id=job_id,revision=revision,window=window,snapshot_ids=[],utterances=[],
                                 line_claims=[],protected=[],same_line=_own.significant_overlap,
                                 claimant=claimant or _lines.new_claimant(key),fence=fence)
    report['beta_line_ledger']={'version':_lines.VERSION,'window_key':key,'mode':'no-dialogue',
                                'status':'committed' if not (problems or races) else 'canonical_arrived','races':races}
    return not (problems or races)


def schedule_within_limit(limit):
    """v3: schedule_around_protection, plus a timing overrun for any take ending after ``limit`` (the start of a
    committed window that rendered without it). The beta reschedule then keeps that cue original, exactly like
    any other overrun; canonical reused lines drop their timing issues afterwards as before."""
    def schedule(cues,durations,*,origin,end,protected):
        placements,problems=schedule_around_protection(cues,durations,origin=origin,end=end,protected=protected)
        flagged={i.get('cue') for i in problems}
        for cue,placement in zip(cues,placements):
            if placement['end']>limit+.0001 and cue['id'] not in flagged:
                placement['timing_overrun']=True
                problems.append({'kind':'speech_timing_overrun','cue':cue['id'],'seconds':placement['end']-placement['start'],
                                 'available_seconds':max(0.,limit-placement['start']),
                                 'limit':'start of a committed window rendered without this take','limit_seconds':limit})
        return placements,problems
    return schedule


def protected_spans(policy):
    return sorted((row['start'],row['end']) for row in (policy or {}).get('protected_intervals',()))


def schedule_with_silence_slack(schedule,*,report,all_cues,analysis_origin,analysis_end,window_limit,canonical):
    """Apply measured tail bounds inside the existing protection/reschedule path."""
    def bounded(cues,durations,*,origin,end,protected):
        placements,issues=schedule(cues,durations,origin=origin,end=end,protected=protected)
        canonical_ids={item.get('cue',item.get('id')) for item in canonical}
        for index,(cue,duration) in enumerate(zip(cues,durations)):
            placement=placements[index]
            obstacles=[dict(p,cue_id=c['id']) for c,p in zip(cues,placements)]
            obstacles.extend(dict(item,cue_id=item.get('cue',item.get('id'))) for item in canonical)
            metadata=_units.silence_slack(cue,raw_turns=report.get('raw_turns') or (),
                transcript=report.get('transcript') or (),all_cues=all_cues,protected=protected,placements=obstacles,
                origin=analysis_origin,analysis_end=analysis_end,window_limit=window_limit)
            placement['timing_slack']=metadata
            if cue['id'] in canonical_ids:continue   # canonical placement remains the committed placement
            allowed=metadata['allowed_end']
            if placement['end']<=allowed+.0001:continue
            # Preserve the existing early-start allowance and prior-take gap.
            # A narrower slot can move a take earlier; it never crosses a wall.
            floor=max(origin,placements[index-1]['end']+.04) if index else origin
            if not any(i.get('cue')==cue['id'] for i in issues):
                try:
                    retry,problems=schedule([cue],[duration],origin=floor,end=min(end,allowed+.04),protected=protected)
                except ValueError:
                    retry,problems=[],[{'kind':'speech_timing_overrun'}]
                if retry and not problems and retry[0]['end']<=allowed+.0001:
                    placements[index]={**retry[0],'timing_slack':metadata}
                    continue
            placement['timing_overrun']=True
            if not any(i.get('cue')==cue['id'] for i in issues):
                issues.append({'kind':'speech_timing_overrun','cue':cue['id'],'seconds':duration,
                    'available_seconds':max(0.,allowed-floor),'timing_slack':metadata})
        return placements,issues
    return bounded


def dialogue_line_fields(cue):
    fields=_ctf.line_fields(cue)
    if cue.get('dialogue_unit'):
        fields.update(source_cue_ids=list(cue['source_cue_ids']),dialogue_unit=copy.deepcopy(cue['dialogue_unit']))
    return fields


def schedule_around_protection(cues,durations,*,origin,end,protected):
    """Schedule generated lines between protected original spans.

    Protected spans plus their transition and the normal line gap are hard
    walls: a line that cannot fit before the next wall is reported as an
    overrun by schedule_lines, never moved across or truncated into it.
    Without protection this is exactly the previous schedule_lines call.
    """
    if not protected:return schedule_lines(cues,durations,origin=origin,end=end)
    pad=PROTECTION_TRANSITION_SECONDS+PROTECTION_SCHEDULE_GAP_SECONDS
    segments=[];cursor=origin
    for begin,stop in protected:
        segments.append((cursor,begin-pad,begin,stop))
        cursor=max(cursor,stop+pad)
    segments.append((cursor,end,end,end))
    placements=[None]*len(cues);issues=[]
    assigned=set()
    previous_stop=float('-inf')
    for floor,limit,wall_start,wall_stop in segments:
        members=[i for i,c in enumerate(cues) if i not in assigned and c['start']<wall_start]
        for i in members:
            if cues[i]['end']>wall_start or cues[i]['start']<previous_stop:
                raise ValueError('A synthesis cue overlaps protected original audio')
        assigned.update(members)
        previous_stop=wall_stop
        if not members:continue
        rows,problems=schedule_lines([cues[i] for i in members],[durations[i] for i in members],origin=floor,end=limit)
        for i,row in zip(members,rows):placements[i]=row
        issues.extend(problems)
    if len(assigned)!=len(cues):raise ValueError('A synthesis cue was not scheduled')
    return placements,issues


def reference_unavailable_issues(ident,cues,reason):
    """Scope a missing actor reference to every affected selected cue.

    The kind stays non-waivable in review. Cue scope lets an explicit
    original-audio policy evaluate a short cue without clearing longer cues
    for the same speaker. No selected cue leaves an unscoped blocker.
    """
    affected=[c for c in cues if c.get('speaker')==ident]
    if not affected:
        return [{'kind':'actor_reference_unavailable','speaker':ident,'reason':reason}]
    return [{'kind':'actor_reference_unavailable','speaker':ident,'cue':c['id'],
             'source_start':c['start'],'source_end':c['end'],'reason':reason,
             'provenance':'actor_reference.make_reference'} for c in affected]


def dominant_reference_label(cue,speech_regions):
    """Raw label with uniquely greatest total overlap with the cue's word spans."""
    words=cue.get('words')
    if not isinstance(words,(list,tuple)) or not words:
        raise ValueError('reference_not_one_raw_speaker')
    scores={}
    try:
        for word in words:
            a,b=float(word['start']),float(word['end'])
            if not math.isfinite(a) or not math.isfinite(b) or a>=b:
                raise ValueError('reference_not_one_raw_speaker')
            for turn in speech_regions:
                left,right,label=float(turn['start']),float(turn['end']),turn['speaker']
                if not math.isfinite(left) or not math.isfinite(right) or left>=right or not isinstance(label,str) or not label:
                    raise ValueError('reference_not_one_raw_speaker')
                amount=max(0.,min(b,right)-max(a,left))
                if amount:scores[label]=scores.get(label,0.)+amount
    except (KeyError,TypeError,ValueError,OverflowError) as error:
        raise ValueError('reference_not_one_raw_speaker') from error
    ranked=sorted(scores.items(),key=lambda item:(-item[1],item[0]))
    if not ranked or (len(ranked)>1 and math.isclose(ranked[0][1],ranked[1][1],rel_tol=0.,abs_tol=1e-9)):
        raise ValueError('reference_not_one_raw_speaker')
    return ranked[0][0]


def line_reference_plan(cue,speech_regions,*,origin,end,rate,limit,allow_extension=True):
    """A source-only span plan; labels describe raw turns, never accepted actor identity.

    The cue's word span bounds its own speech and selects its dominant raw label.
    Short references may extend only within ONE continuous turn of that label,
    stopping at every other-label turn; merged local labels forbid extension.
    Changed spans use reference-only conditioning because their complete ASR is unavailable.
    """
    begin=max(float(cue['start']),float(cue.get('speech_start') or cue['start']))
    stop=min(float(cue['end']),float(cue.get('speech_end') or cue['end']))
    if not all(math.isfinite(x) for x in (begin,stop,origin,end,limit)) or not origin<=begin<stop<=end:
        raise ValueError('reference_invalid_source_span')
    if cue.get('overlapped') or cue.get('turn_overlap') is True:
        raise ValueError('reference_overlapping_dialogue')
    dominant=dominant_reference_label(cue,speech_regions)
    turns=[t for t in speech_regions if min(float(t['end']),stop)>max(float(t['start']),begin)]
    labels={t.get('speaker') for t in turns}
    if not labels or None in labels:
        raise ValueError('reference_not_one_raw_speaker')
    a=math.ceil((begin-origin)*rate-1e-7);b=math.floor((stop-origin)*rate+1e-7)
    own=(a,b)
    cap=math.floor(float(limit)*rate);minimum=math.ceil(rate)
    if labels!={dominant}:
        # A sub-threshold overlap must not create a new whole-cue veto. Cut a
        # clean contiguous subset, without crossing any other raw speaker turn.
        clean=[]
        for label in (dominant,):
            pieces=[(begin,stop)]
            for other in turns:
                if other.get('speaker')==label:continue
                left,right=float(other['start']),float(other['end'])
                kept=[]
                for low,high in pieces:
                    if right<=low or left>=high:kept.append((low,high));continue
                    if left>low:kept.append((low,left))
                    if right<high:kept.append((right,high))
                pieces=kept
            for low,high in pieces:
                if not any(t.get('speaker')==label and min(high,float(t['end']))>max(low,float(t['start'])) for t in turns):continue
                x=math.ceil((low-origin)*rate-1e-7);y=math.floor((high-origin)*rate+1e-7)
                if y-x>=minimum:clean.append((x,y))
        if not clean:raise ValueError('reference_too_short')
        a,b=min(clean,key=lambda span:(-(span[1]-span[0]),span[0]))
    b=min(b,a+cap)
    raw_turn=None
    if b-a<minimum:
        if cap<minimum or not allow_extension:raise ValueError('reference_too_short')
        containers=[t for t in turns if t.get('speaker')==dominant
                    and float(t['start'])<=begin and float(t['end'])>=stop]
        candidates=[]
        for turn in containers:
            lo=max(origin,float(turn['start']));hi=min(end,float(turn['end']))
            for other in speech_regions:
                if other.get('speaker')==turn['speaker']:continue
                left,right=float(other['start']),float(other['end'])
                if right<=begin:lo=max(lo,right)
                elif left>=stop:hi=min(hi,left)
                elif min(right,hi)>max(left,lo):lo=hi
            low=math.ceil((lo-origin)*rate-1e-7);high=math.floor((hi-origin)*rate+1e-7)
            if high-low<minimum:continue
            need=minimum-(b-a)
            x=max(low,a-need//2);y=min(high,x+minimum)
            x=max(low,y-minimum)
            if x<=a and y>=b and y-x>=minimum:
                candidates.append((a-x+y-b,x,y,turn))
        if not candidates:raise ValueError('reference_too_short')
        _,a,b,turn=min(candidates,key=lambda item:(item[0],item[1],str(item[3]['speaker'])))
        raw_turn={k:turn[k] for k in ('start','end','speaker')}
    if b-a<minimum:raise ValueError('reference_too_short')
    interval=[origin+a/rate,origin+b/rate]
    reference_labels={t.get('speaker') for t in speech_regions
                      if min(float(t['end']),interval[1])>max(float(t['start']),interval[0])}
    if reference_labels!={dominant}:raise ValueError('reference_not_one_raw_speaker')
    exact=(begin==float(cue['start']) and stop==float(cue['end']) and (a,b)==own)
    basis=('reference-only-unclear' if cue.get('unclear') else
           'exact-cue-asr' if exact else 'reference-only-expanded-or-trimmed')
    return {'version':LINE_REFERENCE_VERSION,'cue_id':cue['id'],'cue_speech_span':[begin,stop],
            'source_interval':interval,'raw_turn':raw_turn,'transcript_basis':basis,
            'reference_text':cue['text'] if basis=='exact-cue-asr' else '', 'sample_span':[a,b]}


def make_line_reference(speech,cue,speech_regions,origin,rate,path,*,limit,allow_extension=True):
    """Write one untouched channel from this cue's isolated speech stem."""
    import numpy as np
    import soundfile as sf
    plan=line_reference_plan(cue,speech_regions,origin=origin,end=origin+len(speech)/rate,rate=rate,limit=limit,
                             allow_extension=allow_extension)
    a,b=plan['sample_span'];excerpt=speech[a:b]
    if not np.all(np.isfinite(excerpt)):raise ValueError('reference_nonfinite_audio')
    channel=int(np.argmax(np.sum(np.square(excerpt,dtype=np.float64),axis=0)))
    audio=excerpt[:,channel]
    checked=validate_reference_audio(audio,rate,limit=limit)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    sf.write(path,audio,rate,subtype='FLOAT')
    return {'text':plan['reference_text'],'intervals':[plan['source_interval']],
            'seconds':checked['seconds'],'channel':channel,'line_reference':plan}


def scope_bind_problems(problems,cues):
    """SpeakerBank.bind diagnostics (speaker_ambiguous, speaker_embedding_migration_required,
    speaker_reference_mismatch, speaker_embedding_invalid, ...) are identity uncertainty of ONE local speaker. Each
    becomes a cue-scoped speaker_unresolved issue on every affected cue (those cues keep original audio; no identity
    is invented) instead of a window-wide blocker. A problem that names no local speaker, or whose speaker has no
    affected cue, stays an explicit unscoped blocker. ``cues`` are the selected synthesis cues (local labels).
    The full diagnostics stay in report['speaker_bind_problems']."""
    out=[]
    for problem in problems:
        label=problem.get('local_speaker')
        affected=[c for c in cues if label is not None and c.get('speaker')==label]
        if not affected:
            out.append(problem);continue
        out.extend({'kind':'speaker_unresolved','cue':c['id'],'bind_problem':problem.get('kind'),
                    'local_speaker':label,'source_start':c['start'],'source_end':c['end'],
                    **({'detail':problem['detail']} if problem.get('detail') else {})} for c in affected)
    return out


LAYOUT_WAV_MASKS={layout:mask for mask,layout in WAV_MASK_LAYOUTS.items()}


def write_layout_wav(path,samples,rate,layout):
    """Write float PCM; six-channel output carries its explicit speaker mask.

    Mono and stereo keep the previous libsndfile output. 5.1 and 5.1(side)
    are written as WAVE_FORMAT_EXTENSIBLE IEEE float with dwChannelMask 0x3F
    or 0x60F so a later decoder cannot assume the other six-channel order.
    """
    import struct
    import numpy as np
    import soundfile as sf
    samples=np.asarray(samples)
    if samples.ndim!=2 or samples.shape[1]!={'mono':1,'stereo':2,'5.1':6,'5.1(side)':6}.get(layout):
        raise ValueError('Output PCM does not match its explicit channel layout')
    if samples.shape[1]<=2:
        sf.write(path,samples,rate,subtype='FLOAT');return
    data=np.ascontiguousarray(samples,dtype='<f4').tobytes()
    channels=samples.shape[1];align=channels*4
    fmt=struct.pack('<HHIIHH',0xFFFE,channels,rate,rate*align,align,32)
    fmt+=struct.pack('<HHI',22,32,LAYOUT_WAV_MASKS[layout])+bytes.fromhex('0300000000001000800000aa00389b71')
    fact=b'fact'+struct.pack('<II',4,len(samples))
    body=b'WAVE'+b'fmt '+struct.pack('<I',len(fmt))+fmt+fact+b'data'+struct.pack('<I',len(data))+data
    if len(body)+8>0xFFFFFFFF:raise ValueError('Window output exceeds the RIFF size limit')
    temp=Path(str(path)+'.tmp')
    temp.write_bytes(b'RIFF'+struct.pack('<I',len(body))+body)
    os.replace(temp,path)


class PreparedPlanRequired(Exception):
    """Internal scheduler signal: defer this render until fresh reference work can run serially."""


class ActorPipeline:
    def __init__(self,config,*,analysis_service=None,local_cpu_analysis=False):
        """No process-wide CUDA masking: this process loads no analysis model. The movie path injects the shared
        AnalysisService (set_analysis_service). local_cpu_analysis=True is the explicit CPU opt-in for tooling."""
        self.config=config
        self.separator=self.language=self.speakers=None
        self.renderer=None
        self.require_prepared_plan=False
        self.overlap_separator=None
        self.analysis_service=None
        self.local_cpu_analysis=local_cpu_analysis is True
        self._local_language=self._local_speakers=None
        if analysis_service is not None:self.set_analysis_service(analysis_service)

    def set_analysis_service(self,service):
        """Store the shared AnalysisService. Clients are refreshed for the CURRENT owner before every inference
        (_language_client/_speaker_client), including readback and reference-embedding calls."""
        if not callable(getattr(service,'language',None)) or not callable(getattr(service,'speakers',None)):
            raise MovieConfigurationError('analysis_service_invalid',
                                          'AnalysisService must provide language(owner) and speakers(owner)')
        self.analysis_service=service
        self.language=self.speakers=None
        return service

    def _no_service(self):
        return MovieConfigurationError('analysis_service_missing',
            'The movie path needs the injected AnalysisService (ActorPipeline.set_analysis_service); local CPU '
            'analysis is explicit opt-in only (ActorPipeline(config, local_cpu_analysis=True))')

    def _language_client(self,owner):
        if self.analysis_service is not None:
            self.language=self.analysis_service.language(owner)
        elif self.local_cpu_analysis:
            if self._local_language is None:
                from .language import LanguageTools
                self._local_language=LanguageTools(self.config,device='cpu')
            self.language=self._local_language
        else:raise self._no_service()
        return self.language

    def _speaker_client(self,owner):
        if self.analysis_service is not None:
            self.speakers=self.analysis_service.speakers(owner)
        elif self.local_cpu_analysis:
            if self._local_speakers is None:
                from .speakers import SpeakerAnalyzer
                self._local_speakers=SpeakerAnalyzer(self.config,cinematic=False,device='cpu',cache=None)
            self.speakers=self._local_speakers
        else:raise self._no_service()
        return self.speakers

    def _speaker_policy(self,owner):
        """The backend's EFFECTIVE clustering/embedding/device policy (SpeakerAnalyzer.policy_metadata()); never a
        constant of this module. Missing metadata is a configuration fault, not an assumed policy."""
        client=self._speaker_client(owner)
        report=getattr(client,'policy_metadata',None)
        policy=report() if callable(report) else None
        if not isinstance(policy,dict) or not policy:
            raise MovieConfigurationError('speaker_policy_unreported',
                                          'The speaker backend did not report policy_metadata()')
        return policy

    def _device_report(self,owner):
        if self.analysis_service is not None:
            report=getattr(self.analysis_service,'device_report',None)
            return report() if callable(report) else None
        return {'backend':'local-cpu-opt-in',
                'language':self._language_client(owner).device_report(),
                'speakers':self._speaker_client(owner).device_report()}

    def _speech(self,owner,wav,language=None,*,vad=True,beam_size=5):
        """ASR through the current owner's client, called directly. The ONE authoritative raw-inference cache
        (exact content + model identity) lives in the AnalysisService's GPU child; this process adds none."""
        segments,info=self._language_client(owner).speech(wav,language,vad=vad,beam_size=beam_size)
        return segments,info

    def _correct_languages(self,owner,analysis_wav,recognition,segments,language,folder):
        """(segments, corrections, spans): segments whose AUDIO language differs from the window's (>= LID_MIN_PROBABILITY)
        are decoded again in that language. See apply_language_corrections."""
        eligible=[i for i,seg in enumerate(segments) if float(seg['end'])-float(seg['start'])>=LID_MIN_SECONDS
                  and str(seg.get('text','')).strip()]
        if not eligible:return segments,[],[]
        import soundfile as sf
        total=len(recognition)/16000.
        raw=self._language_client(owner).language_id(analysis_wav,[[max(0.,float(segments[i]['start'])-LID_PAD_SECONDS),
                                                                  min(total,float(segments[i]['end'])+LID_PAD_SECONDS)] for i in eligible])
        ids=[None]*len(segments)
        for i,ident in zip(eligible,raw):ids[i]=ident
        def redecode(i,found):
            seg=segments[i];a=max(0.,float(seg['start'])-LID_PAD_SECONDS);b=min(total,float(seg['end'])+LID_PAD_SECONDS)
            path=folder/('language-redecode-%d.wav'%i)
            sf.write(path,recognition[round(a*16000):round(b*16000)],16000,subtype='PCM_16')
            recovered,_=self._speech(owner,path,found,vad=False)
            return shift_segments(recovered,a)
        return apply_language_corrections(segments,ids,redecode,language['language'])

    def _nonverbal_audio(self,owner,speech,rate,origin,cues,folder):
        """Marks cues whose ORIGINAL audio is not speech to dub: singing or chant, coughs, sneezes, sighs, screams without
        words, silence the recognizer invented text over, and audio that is English in a foreign film. Evidence: the
        language of the audio (cue['language']), pitch and level measured on the separated vocals, and AudioSet labels from
        the analysis child (when audio_event_model is set). See relay/nonverbal.py for the rules and thresholds."""
        import soundfile as sf
        pending=[cue for cue in cues if not cue.get('unclear')]
        if not pending:return {}
        mono=_evidence.to_mono_16k(speech,rate)
        wav=folder/'vocals-16k.wav'
        sf.write(wav,mono,16000,subtype='FLOAT')
        spans=[[max(0.,float(cue['start'])-origin),float(cue['end'])-origin] for cue in pending]
        try:events=self._language_client(owner).audio_events(wav,spans)
        except (MovieConfigurationError,MovieContentFallback):events=[None]*len(pending)
        decided={}
        for cue,span,labels in zip(pending,spans,events):
            a,b=int(span[0]*16000),int(span[1]*16000)
            features=nonverbal_evidence(mono[a:b],16000,labels) if b>a else {}
            reason=nonverbal_reason(cue.get('text'),features,labels,duration=float(cue['end'])-float(cue['start']))
            record={'features':features,'events':labels,'language':cue.get('language')}
            if features.get('f0_median_hz'):cue['f0_median_hz']=features['f0_median_hz']       # lets a borrowing line pick a similar voice
            if reason:
                cue['unclear']=True;cue['nonverbal']=reason;record['reason']=reason
            decided[cue['id']]=record
        return decided

    def _english(self,owner,text,language):
        """Translation through the current owner's client, called directly (cached in the GPU child)."""
        english,basis=self._language_client(owner).english(text,language)
        return english,basis

    def _embedding(self,owner,samples):
        """One speaker embedding (list of floats) or None when the backend reports insufficient voice evidence."""
        import math
        try:vector=list(map(float,self._speaker_client(owner).embed(samples)))
        except MovieContentFallback:return None
        if not vector or not all(math.isfinite(x) for x in vector):return None
        return vector

    def close(self):
        """The AnalysisService is shared and owned by the server; only this pipeline's actor renderer closes."""
        if self.renderer:self.renderer.close()

    def protection_ledger(self,store,job_id,revision,source_identity,layout):
        """Cross-window kept-original ledger, or None without a source identity."""
        if source_identity is None:return None
        scope=protection_scope(source_fingerprint=source_identity.get('source_fingerprint'),
            audio_index=source_identity.get('audio_index'),channel_layout=layout,
            engine_fingerprint=self.config.engine_fingerprint(),
            policy_version=ORIGINAL_POLICY_VERSION+'+'+NATURAL_POLICY_VERSION)
        return ProtectionLedger(store,job_id,revision,scope)

    def process(self,source_wav,origin,captions,folder,store,job_id,owner,*,render_start=None,render_end=None,revision=1,
                **options):
        """v4 (caption_targets): a window renders only under the coordinator's chunk claim when its job exists in the
        store (W2 P3-1: no silent fence skip), registers its intent before preparation (W2 P2-2) and releases its
        claims and intent on every exit. Without caption_targets this is the unchanged pipeline."""
        if not _ctf.enabled(self.config):
            return self._process(source_wav,origin,captions,folder,store,job_id,owner,render_start=render_start,
                                 render_end=render_end,revision=revision,**options)
        import sqlite3
        import soundfile as sf
        fence=chunk_fence(store,job_id,revision)
        if fence is None and store.job(job_id) is not None:
            raise ValueError('caption_targets: a window of a stored job renders only under its coordinator chunk claim '
                             '(pass the coordinator._ClaimedStore); refusing to skip the fence')
        info=sf.info(str(source_wav))
        end=origin+info.frames/info.samplerate
        rs=origin if render_start is None else render_start
        re_=end if render_end is None else render_end
        claimant=_lines.new_claimant(_lines.window_key(rs,re_))
        _lines.register_window(store,job_id=job_id,revision=revision,claimant=claimant,render_range=[rs,re_],
                               analysis_range=[origin,end],fence=fence)
        try:
            return self._process(source_wav,origin,captions,folder,FencedStore(store,fence) if fence else store,job_id,
                                 owner,render_start=render_start,render_end=render_end,revision=revision,
                                 claimant=claimant,fence=fence,**options)
        finally:
            try:_lines.release(store,job_id=job_id,revision=revision,claimant=claimant,reason='window_finished')
            except sqlite3.Error:pass        # unreleased rows expire with their fence, lease or dead holder

    def _process(self,source_wav,origin,captions,folder,store,job_id,owner,*,render_start=None,render_end=None,revision=1,
                 prepare_only=False,source_channel_layout=None,source_identity=None,subtitle_choice=None,claimant=None,
                 fence=None,final_window=False):
        """``source_channel_layout`` is the selected ffprobe stream (or its layout).

        ``final_window`` (keyword, default False): this is the movie's LAST chunk. Only then, when the decoded audio
        ends inside the render range (render_start < decoded end < render_end, e.g. the audio stream ends before the
        container duration), the render range ends at the exact decoded end (report['render_end']); nothing is padded,
        shifted or invented. Every other window whose decoded audio is short still fails.

        The decoded WAV's own channel mask is also checked. Six channels need
        explicit 5.1 or 5.1(side) evidence; unknown or contradictory evidence
        fails here, before separation, recognition or synthesis.

        ``source_identity`` ({'source_fingerprint', 'audio_index'}) enables the
        persisted kept-original ledger shared by every window and seek of this
        job revision. Without it protection stays window-local and the report
        says so (``cross_window_persistence: False``).
        """
        import numpy as np
        import soundfile as sf
        import torch
        from scipy.signal import resample_poly
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        started=time.perf_counter()
        original,rate=sf.read(source_wav,dtype='float32',always_2d=True)
        self._admission_window=_beta.admission_window(self.config,job_id=job_id,revision=revision,origin=origin,
            rate=rate,source_identity=source_identity,folder=folder,source_wav=source_wav)
        if rate!=48000:raise ValueError('Actor pipeline needs the original 48 kHz channel layout')
        if original.shape[1] not in (1,2,6):raise ValueError('Resolve the channel layout before dubbing')
        layout_evidence=resolve_channel_layout(original.shape[1],source=source_channel_layout,wav=wav_format(source_wav))
        layout=layout_evidence['layout']
        lfe=3 if original.shape[1]==6 else None
        end=origin+len(original)/rate
        render_start=origin if render_start is None else render_start
        render_end=end if render_end is None else render_end
        if final_window is True and render_start<end<render_end:render_end=end   # exact decoded geometry, last chunk only
        if not origin<=render_start<render_end<=end:raise ValueError('Render range is outside the analyzed audio')
        first_pass=removal=None;fp_scope=fp_wall=None
        line_voice_mode=_beta.beta_settings(self.config) is not None
        if _fpf.enabled(self.config):
            # C's validated one-pass Demucs bundle: processing STEREO is the working original, raw vocals feed
            # ASR/speakers/references/pan, removal stem feeds ONLY the mix; source lineage stays separate.
            from . import first_pass_parent as _first_pass
            fp_scope=_subs.plan(self.config,original,rate,origin,captions,lfe,None)
            fp_started=time.perf_counter()
            first_pass=_first_pass.run(self.config,owner,folder=folder,original_source_wav=source_wav,
                source_layout_evidence=layout_evidence,origin=origin,frames=len(original),scope=fp_scope)
            fp_wall=time.perf_counter()-fp_started
            if first_pass is None:raise ValueError('demucs_first_pass is enabled but no first-pass bundle was returned')
            fp_arrays=_fpf.arrays(first_pass,len(original),rate)
            source_layout_evidence=layout_evidence
            original,speech,removal=fp_arrays['processing'],fp_arrays['raw_vocals'],fp_arrays['removal_stem']
            layout_evidence=_fpf.layout_evidence(first_pass);layout='stereo';lfe=None
            if self._admission_window is not None:
                self._admission_window.update(
                    source_wav={'path':str(Path(first_pass['processing']['wav']).resolve()),'sha256':first_pass['processing']['sha256']},
                    speech_stem={'path':str(Path(first_pass['raw_vocals']['wav']).resolve()),'sha256':first_pass['raw_vocals']['sha256']},
                    original_source=first_pass['original_source'],first_pass_key=first_pass['key'])
        caption_inputs=_ctf.caption_inputs(line_voice_config(self.config),folder,captions,subtitle_choice,origin,end)
        # beta.5 fix switches: Whisper's two-part no-speech rule in the first-pass lane;
        # caption-edge pause split and start-owned caption issues with caption_targets.
        whisper_rule=first_pass is not None
        caption_rules=_ctf.enabled(self.config)
        edge_pause=CAPTION_EDGE_PAUSE_SECONDS if caption_rules else None
        # v2: channel-aware ASR input (first pass). v3: one cue-start ownership rule with boundary
        # original-audio fallback, placement floor and the atomic line ledger (caption_targets).
        if caption_rules:ownership_modules_verified()
        fix_record=({'version':BETA5_ASR_FIX_VERSION,'whisper_no_speech':whisper_rule,'caption_edge_pause':edge_pause,
                     'caption_issue_owner':'speech-start' if caption_rules else 'overlap',
                     'asr_input_policy':ASR_INPUT_POLICY if whisper_rule else None,
                     'dialogue_ownership':{'version':_own.VERSION,'line_ledger':_lines.VERSION,
                                           'modules':dict(OWNERSHIP_MODULE_SHA256)} if caption_rules else None}
                    if whisper_rule or caption_rules else None)
        if self._admission_window is not None and caption_inputs is not None:self._admission_window['caption_inputs']=caption_inputs
        ledger=DialogueLedger(store,job_id,revision)
        canonical=ledger.between(origin,end)
        protection=self.protection_ledger(store,job_id,revision,source_identity,layout)
        pcm_identity=hashlib.sha256(np.ascontiguousarray(original).tobytes()).hexdigest()
        plan_path=folder/'actor-prepared-plan.json'
        stem_path=folder/'speech-estimate.wav'
        key_parts=[pcm_identity,rate,list(original.shape),origin,render_start,
            render_end,revision,self.config.engine_fingerprint(),captions,layout,
            # The configured speaker policy (cheap, no model); the effective policy itself is in the report.
            {'speaker_clustering_policy':getattr(self.config,'data',self.config).get('speaker_clustering_policy','community1-stock'),
             'analysis_backend':'analysis-service' if self.analysis_service is not None else 'local-cpu'}]
        if protection:key_parts.append(protection.key)
        if first_pass is not None:key_parts.append(_fpf.record(first_pass))
        if caption_inputs is not None:key_parts.append(_ctf.key_part(caption_inputs))
        if fix_record is not None:key_parts.append(fix_record)
        unit_policy=({'version':_units.POLICY_VERSION,'helper_sha256':file_hash(Path(_units.__file__)),
                      'voice_policy':VOICE_POLICY_VERSION} if line_voice_mode else None)
        if unit_policy is not None:key_parts.append(unit_policy)
        plan_key=hashlib.sha256(json.dumps(key_parts,sort_keys=True).encode()).hexdigest()
        prepared={}
        if plan_path.is_file():
            try:prepared=json.loads(plan_path.read_text(encoding='utf-8'))
            except (ValueError,OSError):prepared={}
        if prepared.get('key')==plan_key and (first_pass is not None or stem_path.is_file()):
            known=store.speakers(job_id)
            required_voice_ids={c['speaker'] for c in prepared.get('cues') or ()}
            unchanged=all((line_voice_mode and voice.get('line_reference',{}).get('version')==LINE_REFERENCE_VERSION
                           or known.get(ident,{}).get('voice')==voice) and
                (file_hash(voice['reference'])==voice['reference_sha256'] if voice.get('reference') else
                 ident not in required_voice_ids and not ident.startswith('line-'))
                for ident,voice in prepared['voices'].items())
            if unchanged and (_fpf.prepared_matches(prepared,first_pass) if first_pass is not None
                              else file_hash(stem_path)==prepared.get('speech_sha256')):
                if first_pass is None:speech,stem_rate=sf.read(stem_path,dtype='float32',always_2d=True)
                else:stem_rate=rate
                if speech.shape!=original.shape or stem_rate!=rate:raise ValueError('Prepared dialogue geometry changed')
                if prepared.get('channel_layout')!=layout:raise ValueError('Prepared channel layout changed')
                _validate_policy(prepared.get('original_audio_policy'),origin=origin,end=end,
                    beta_original_fallback=prepared['report'].get('beta_original_fallback'),beta_enabled=_beta.beta_settings(self.config) is not None,
                    synthesis_cues=prepared['cues']+ownership_accounted(prepared['report'],prepared.get('all_cues')),
                    channel_layout=layout,channels=original.shape[1],
                    all_cues=prepared.get('all_cues'),speech_regions=prepared['report'].get('raw_turns'),
                    render_start=render_start,render_end=render_end)
                report=prepared['report'];report['preparation_reused']=True;report['memory_stages']=[]
                if self._admission_window is not None:
                    # The effective backend policy recorded when this plan was analysed (receipts stay honest).
                    self._admission_window['speaker_policy']=report.get('speaker_policy')
                owner.check()
                if prepare_only:
                    report.update(state='prepared',elapsed_seconds=time.perf_counter()-started)
                    if fence is not None:_lines.check_fence(store,fence)
                    return report
                return self._render_prepared(original,speech,rate,origin,end,prepared['cues'],prepared['all_cues'],
                    prepared['voices'],folder,store,job_id,owner,revision,report,started,protection=protection,removal=removal,
                    claimant=claimant,fence=fence)
        owner.check()
        if self.require_prepared_plan:
            # The scheduler permitted this render alongside another window's preparation only because the actor
            # plan was committed. A stale plan must not silently enter binding/reference admission in parallel.
            raise PreparedPlanRequired('Prepared actor plan is no longer reusable; defer fresh reference work')
        if first_pass is None:
            if file_hash(self.config['cinematic_checkpoint'])!=CHECKPOINT_SHA256:
                raise ValueError('Cinematic checkpoint changed')
            separator_sources=Path(self.config['cinematic_code_root'])/'src'
            # WAV PEAK metadata can contain a write timestamp; unchanged decoded
            # samples still represent the same inference input.
            identity=[pcm_identity,rate,list(original.shape),CHECKPOINT_SHA256,file_hash(Path(__file__).with_name('cinematic.py')),
                [(str(p.relative_to(separator_sources)),file_hash(p)) for p in sorted(separator_sources.rglob('*.py'))],
                int(self.config['cpu_threads']),torch.__version__,5]
            scope=_subs.plan(self.config,original,rate,origin,captions,lfe,self.separator)
            if scope is not None:identity.append(scope['key'])
            separation_key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            stem_path=folder/'speech-estimate.wav';stem_receipt=folder/'speech-estimate.json'
            cached=json.loads(stem_receipt.read_text()) if stem_receipt.is_file() else {}
            if cached.get('key')==separation_key and stem_path.is_file() and file_hash(stem_path)==cached.get('sha256'):
                speech,stem_rate=sf.read(stem_path,dtype='float32',always_2d=True)
                if speech.shape!=original.shape or stem_rate!=rate:raise ValueError('Cached dialogue geometry changed')
                separation={**cached['metrics'],'original_compute_seconds':cached['metrics']['seconds'],'seconds':0,'cache_reused':True}
            else:
                if self.separator is None:
                    self.separator=CinematicSeparator(self.config['cinematic_checkpoint'],self.config['cinematic_code_root'],
                        threads=min(8,int(self.config['cpu_threads'])),batch_channels=5)
                speech,separation=_subs.separate(self.separator,original,rate,origin,lfe,owner,scope)
                sf.write(stem_path,speech,rate,subtype='FLOAT')
                atomic_json(stem_receipt,{'key':separation_key,'sha256':file_hash(stem_path),'metrics':separation})
        else:
            separation=_fpf.separation(first_pass,fp_scope,fp_wall)
        channels=[i for i in range(original.shape[1]) if i!=lfe]
        asr_input=None
        if first_pass is None:
            # Use the strongest dialogue channel, avoiding phase cancellation in a
            # downmix. Independent off-axis/overlapping lines remain review cases.
            analysis=speech[:,max(channels,key=lambda c:float(np.mean(speech[:,c]**2))) ]
            recognition=resample_poly(analysis,1,3).astype('float32')
        else:
            # The processing downmix put centre dialogue at -10.7 dB (5.1 ITU);
            # restore it to unity for ASR and speaker analysis.
            recognition,asr_input=first_pass_recognition(speech,source_layout_evidence.get('layout'),
                (first_pass.get('processing') or {}).get('transform'),local_gain=asr_local_gain_enabled(self.config))
        analysis_wav=folder/'dialogue-16k.wav'
        sf.write(analysis_wav,recognition,16000,subtype='PCM_16')
        owner.check()
        # Raw inference below goes directly through the current owner's AnalysisService clients. The one
        # authoritative raw cache (ASR/MT, diarization, embeddings incl. those inside refine_utterances) is in the
        # GPU child; a rebinding therefore re-runs only the cheap decisions here.
        stamp=time.perf_counter();segments,language=self._speech(owner,analysis_wav)
        language_spans,language_corrections=[],[]
        if overlap_policy(self.config)['language_correction']:
            segments,language_corrections,language_spans=self._correct_languages(owner,analysis_wav,recognition,segments,language,folder)
        timing={'separation':separation['seconds'],'asr':time.perf_counter()-stamp}
        owner.check()
        speaker_policy=self._speaker_policy(owner)
        if self._admission_window is not None:self._admission_window['speaker_policy']=speaker_policy
        bank=SpeakerBank(store,job_id,self.config)
        known_speakers=bank.reference_vectors(lambda samples:self._speaker_client(owner).embed(samples))
        stamp=time.perf_counter();turns,aligned,vectors=self._speaker_client(owner).analyze(analysis_wav,origin)
        rescue_records=[]
        if self.config['asr_caption_gap_rescue']:
            for index,(begin,stop) in enumerate(rescue_windows(segments,turns,captions,origin,end,whisper_no_speech=whisper_rule)):
                owner.check();rescue_started=time.perf_counter()
                path=folder/f'asr-rescue-{index}.wav'
                sf.write(path,recognition[round((begin-origin)*16000):round((stop-origin)*16000)],16000,subtype='PCM_16')
                recovered,_=self._speech(owner,path,language['language'] if language['probability']>=.65 else None,vad=False)
                segments,added=merge_rescue(segments,recovered,begin-origin,whisper_no_speech=whisper_rule)
                rescue_records.append({'start':begin,'end':stop,'added_segments':added,'seconds':time.perf_counter()-rescue_started,
                                       'transcript':recovered,'vad_filter':False})
        identity_units=speaker_units(segments,aligned,captions,origin,whisper_no_speech=whisper_rule)
        owner.check()
        refined,refined_vectors,speaker_evidence=self._speaker_client(owner).refine_utterances(recognition,16000,identity_units,vectors,origin,speech_regions=turns,known_speakers=known_speakers,
            **({'whisper_no_speech':True} if whisper_rule else {}))
        timing['asr_rescue']=sum(r['seconds'] for r in rescue_records)
        timing['speakers']=time.perf_counter()-stamp-timing['asr_rescue']
        selected=[c for c in captions if c['end']>origin and c['start']<end]
        policy_knobs=overlap_policy(self.config)
        cues,issues,discarded=plan_dialogue(segments,refined,selected,origin=origin,speech_regions=turns,
            whisper_no_speech=whisper_rule,caption_edge_pause=edge_pause,minimum_probability=policy_knobs['min_probability'],
            overlap_fraction=policy_knobs['word_fraction'],word_floor=policy_knobs['word_floor'],
            low_word_share=policy_knobs['low_word_share'])
        for cue in cues:
            # Owner rule (2026-10-19): laughter, sound tags, music marks and Whisper's invented credit lines are not speech.
            # They stay original audio: no translation, no voice, nothing removed (handled as an unclear line below).
            reason=nonverbal_reason(cue.get('text')) or sparse_speech_reason(float(cue['end'])-float(cue['start']),len(cue.get('words') or []))
            if reason:cue['unclear']=True;cue['nonverbal']=reason
        window_language=language['language']
        nonverbal_record={}
        for cue in cues:
            cue['language'],cue['language_from_audio']=cue_language(cue,language_spans,origin,window_language)
            if cue['language_from_audio'] and cue['language']=='en' and window_language!='en' and not cue.get('unclear'):
                cue['unclear']=True;cue['nonverbal']='english_audio'      # English spoken in a foreign film: it stays as it is
        if policy_knobs['nonverbal_audio']:
            nonverbal_record=self._nonverbal_audio(owner,speech,rate,origin,cues,folder)
        raw_source_cues=None
        if line_voice_mode:
            cues,overlap_issues=_units.annotate_overlaps(cues,turns,span_seconds=policy_knobs['span_seconds'],span_fraction=policy_knobs['span_fraction'])
            issues.extend(i for i in overlap_issues if i not in issues)
        # Audit the original local labels before any grouping or ownership decision.
        audits={}
        for segment in segments:
            if segment['end']-segment['start']<.5 or is_no_speech(segment,whisper_rule):continue
            absolute={**segment,'start':segment['start']+origin,'end':segment['end']+origin}
            candidates=sorted(((overlap(absolute,t),t['speaker']) for t in refined),reverse=True)
            if not candidates or candidates[0][0]<=0:continue
            a,b=round(segment['start']*16000),round(segment['end']*16000)
            vector=self._embedding(owner,recognition[a:b])
            if vector is not None:audits.setdefault(candidates[0][1],[]).append(vector)
        for label,values in audits.items():
            if label not in {c['speaker'] for c in cues}:continue
            if len(values)>=3:
                scores=[cosine(a,b) for i,a in enumerate(values) for b in values[i+1:]]
                if sum(s<.3 for s in scores)>len(scores)*.4:
                    issues.append({'kind':'possible_merged_speakers','speaker':label,
                                   'minimum_similarity':min(scores),'utterances':len(values)})
        if line_voice_mode:
            raw_source_cues=copy.deepcopy(cues)
            cues,issues=_units.group_units(cues,selected,turns,origin=origin,end=end,
                render_start=render_start,render_end=render_end,
                issues=issues,merged_labels={i['speaker'] for i in issues if i.get('kind')=='possible_merged_speakers'},
                caption_edge_pause=CAPTION_EDGE_PAUSE_SECONDS,fragment_seconds=policy_knobs['fragment_seconds'])
        all_cues=cues
        ownership=None
        if caption_rules:
            # v3 single ownership by speech start: only the window holding a cue's start
            # selects it (B's caption gate rule). Every other window overlapping that cue
            # or a caption it does not own accounts for the span as a boundary item.
            ownership=_own.plan(all_cues,selected,turns,render_start=render_start,render_end=render_end,
                                source_identity=source_identity,caption_text=caption_text)
            owned_ids=set(ownership['owned_cue_ids'])
            cues=[c for c in all_cues if c['id'] in owned_ids]
        else:
            cues=[c for c in all_cues if c['end']>render_start and c['start']<render_end]
        selected_ids={c['id'] for c in cues}
        issues=[i for i in issues if i.get('cue') in selected_ids or
            (i.get('kind')=='possible_merged_speakers' and i.get('speaker') in {c.get('speaker') for c in cues}) or
            (i.get('caption') and keep_caption_issue(i['caption'],render_start,render_end,speech_regions=turns,
                                                     by_speech_start=caption_rules))]
        issues.extend({'kind':'singing_requires_pitch_tempo_path','cue':c['id']} for c in cues if c.get('singing'))
        vectors=refined_vectors
        for cue in cues:
            # A refined utterance does not erase evidence of simultaneous voices.
            overlap_spans=[]
            for index,left in enumerate(turns):
                for right in turns[index+1:]:
                    if left.get('speaker')==right.get('speaker'):continue
                    shared={'start':max(left['start'],right['start']),'end':min(left['end'],right['end'])}
                    if overlap(shared,cue)>0:
                        overlap_spans.append((max(shared['start'],cue['start']),min(shared['end'],cue['end'])))
                    if overlap(shared,cue)>max(policy_knobs['span_seconds'],policy_knobs['span_fraction']*(float(cue['end'])-float(cue['start']))):
                        cue['overlapped']=True
                        issues.append({'kind':'overlapping_dialogue','cue':cue['id'],
                                       'start':max(shared['start'],cue['start']),'end':min(shared['end'],cue['end'])})
            cue['overlap_intervals']=[list(span) for span in merge_spans(overlap_spans)]
            cue['overlap_seconds']=sum(b-a for a,b in cue['overlap_intervals'])
            cue['cue_seconds']=float(cue['end'])-float(cue['start'])
        if policy_knobs['solo_relax']:
            relax_solo_overlap(cues,issues,all_cues)      # each relaxed cue records 'overlap_relaxed' on itself
        # Kept-original spans persisted by any window or earlier pass of this
        # revision bind this window: a selected cue touching one (other than
        # the identical kept cue) is a non-waivable cue-scoped blocker, and
        # scope/revision/layout mismatches are unscoped blockers.
        persisted=protection.between(origin,end,window=(origin,end),original=original,rate=rate) if protection else None
        if persisted:
            issues.extend(persisted['issues'])
            issues.extend(persisted_protection_issues(persisted['active'],cues))
        report={'origin':origin,'end':end,'language':language,'cues':cues,'issues':issues,
                'discarded':discarded,'separation':separation,'timing':timing,'actor_cloning':True,
                'human_listening_acceptance':False,'speaker_evidence':speaker_evidence,
                'raw_turns':turns,'refined_turns':refined,'transcript':segments,'asr_rescue':rescue_records,
                'speaker_analysis_units':[{'start':u['start']+origin,'end':u['end']+origin,'source_segments':u['source_segments']} for u in identity_units],
                'render_start':render_start,'render_end':render_end,
                'channel_layout':layout,'channel_layout_evidence':layout_evidence,
                # Effective backend policy (SpeakerAnalyzer.policy_metadata()) and device report; no assumed clusterer.
                'speaker_policy':speaker_policy,'analysis_devices':self._device_report(owner),
                'persisted_original_protection':None if persisted is None else {
                    'version':persisted['version'],'active':persisted['active'],
                    'stale_revision_rows':len(persisted['stale_revision']),'pcm_evidence':persisted['pcm_evidence']}}
        def save(state):
            report.update(state=state,elapsed_seconds=time.perf_counter()-started)
            from .resource_budget import ResourceBudget
            report.setdefault('memory_stages',[]).append({'stage':state,'ram_free_mib':ResourceBudget.ram_free()})
            fenced_json(store,fence,folder/'actor-receipt.json',report)
            return report
        if first_pass is not None:
            report.update(source_channel_layout=source_layout_evidence.get('layout'),
                          source_channel_layout_evidence=source_layout_evidence,first_pass=_fpf.record(first_pass))
        if fix_record is not None:report['beta5_asr_fix']={**fix_record,'asr_input':asr_input}
        if language_corrections:report['language_corrections']=language_corrections
        if nonverbal_record:report['nonverbal_evidence']=nonverbal_record
        if unit_policy is not None:
            report['dialogue_unit_policy']=dict(unit_policy)
            report['dialogue_raw_cues']=raw_source_cues
        if ownership is not None:
            decisions,_=_own.decide(ownership['items'],canonical,render_start=render_start,render_end=render_end,
                canonical_intervals=_ctf.canonical_intervals,take_span=take_span,evaluated_at='prepare')
            report['beta_dialogue_ownership']={**ownership,'decisions':decisions,'source_identity':source_identity}
        if not cues and not turns and not segments and not canonical:
            # v3: a no-dialogue window still records that it rendered (original audio, nothing carried), so an
            # owner's take never overruns into it unheard; if a take arrived meanwhile it renders to carry it.
            if ownership is None or record_no_dialogue_window(store,job_id,revision,report,origin,end,claimant,fence):
                report.update(audio=str(source_wav),sha256=file_hash(source_wav),mode='no-dialogue')
                return save('ready')
            canonical=ledger.between(origin,end)
        if self.config['overlap_checkpoint'] and any(i['kind']=='overlapping_dialogue' for i in issues):
            report['overlap_diagnostics']=self.inspect_overlaps(recognition,origin,issues,language,vectors,folder,owner)
        # Every gate classifies natural short exceptions over all window cues,
        # complete raw speech regions and the cumulative issue list. Only
        # approved short-original and beta fallback cues leave synthesis;
        # remaining global, singing, clipping and unmapped issues still block.
        # report['cues'] stays the complete selected list.
        active=list(cues);kept_ids=set()
        # Keep the detector's local-label namespace across every policy recompute after actor binding.
        cue_local_speakers={c['id']:c.get('speaker') for c in cues}
        line_voices={}
        def blocked():
            nonlocal active,kept_ids
            policy_issues,fallback=_beta.split_cue_fallbacks(self.config,issues,cue_speakers=cue_local_speakers)
            policy=classify_original_audio(all_cues,turns,policy_issues,origin=origin,end=end,
                render_start=render_start,render_end=render_end,canonical=canonical)
            kept_ids=set(policy['preserved_cue_ids'])|set(fallback)
            if _beta.beta_settings(self.config):
                report['beta_original_fallback']=fallback
                # Retain the detector's full reasons/scores as well as the per-cue classification.
                report['beta_cue_fallback_issues']=[dict(i) for i in issues if i not in policy_issues]
                report['beta_fallback_local_speakers']=dict(cue_local_speakers)
            active=[c for c in cues if c['id'] not in kept_ids]
            # A separate list: later readback/timing issues must not alter the policy receipt.
            report.update(original_audio_policy=policy,issues=list(policy['remaining_issues']),
                          synthesis_cue_ids=[c['id'] for c in active],kept_original_cue_ids=policy['preserved_cue_ids'])
            return bool(policy['remaining_issues'])
        if line_voice_mode:
            report['line_voice_policy']={'version':LINE_REFERENCE_VERSION,
                'voice_policy':VOICE_POLICY_VERSION,'bank_reference_min_seconds':BANK_REFERENCE_MIN_SECONDS,
                'identity_gate':False,'identity_verified':False,
                'overlap_policy':'whole-cue-original-splitting-deferred',
                'caption_target_switch':'actor_beta.caption_targets',
                'caption_targets_for_synthesis':False,'ownership_uses_configured_switch':caption_rules,
                'english_policy':'aligned-per-cue-English-caption-else-clear-source-translation',
                'reference_min_seconds':1.0,'source':'raw_vocals' if first_pass is not None else 'speech-estimate'}
            # Local clustering and bank matching remain evidence, never a substitute
            # for the line's own source sample or a requirement to use that sample.
            identity=[];remaining=[];merged_labels=set()
            for issue in issues:
                mapped_merge=(issue.get('kind')=='possible_merged_speakers' and issue.get('speaker') is not None
                              and issue['speaker'] in set(cue_local_speakers.values()))
                if mapped_merge:merged_labels.add(issue['speaker'])
                if (issue.get('kind')=='speaker_unresolved' and issue.get('cue') in selected_ids) or mapped_merge:
                    identity.append(dict(issue))
                else:remaining.append(issue)
            issues[:]=remaining
            report['line_identity_diagnostics']=identity
            merged_cue_ids={cid for cid,label in cue_local_speakers.items() if label in merged_labels}
            report['line_reference_extension_blocked']={cid:'possible_merged_speakers' for cid in sorted(merged_cue_ids)}
            for cue in all_cues:
                cue['local_speaker']=cue.get('speaker')
            for cue in cues:
                if cue.get('caption'):cue['caption_not_used']='subtitles-are-viewing-only'
                if cue.get('overlapped') or cue.get('turn_overlap') is True:
                    if not any(i.get('kind')=='overlapping_dialogue' and i.get('cue')==cue['id'] for i in issues):
                        issues.append({'kind':'overlapping_dialogue','cue':cue['id']})
                    continue
                if cue.get('singing'):continue
                if cue.get('unclear') and cue.get('translation')!='aligned-English-caption':
                    issues.append({'kind':'uncertain_transcript','cue':cue['id']});continue
        if blocked():return save('needs_review')
        # A word-uncertain cue must reach the caption planner with a chance to bind its local speaker.
        # Only this one fallback is provisional; short-policy and all other fallback reasons stay excluded.
        short_preserved=set(report['original_audio_policy']['preserved_cue_ids'])
        before_caption_fallback=report.get('beta_original_fallback') or {}
        caption_binding_ids={c['id'] for c in cues if caption_inputs is not None and c['id'] not in short_preserved
            and set(before_caption_fallback.get(c['id']) or ())=={'unclear_dialogue_without_aligned_caption'}}
        binding_cues=[c for c in cues if c['id'] not in kept_ids or c['id'] in caption_binding_ids]
        binding_ids={c['id'] for c in binding_cues}
        needed={cue_local_speakers[c['id']] for c in binding_cues if cue_local_speakers.get(c['id']) is not None}
        if line_voice_mode:needed-=merged_labels
        binding_vectors={k:v for k,v in vectors.items() if k in needed}
        mapping,problems=bank.bind(binding_vectors)
        bind_decisions={label:{**value,'candidates':list(value.get('candidates') or ())}
                        for label,value in (getattr(bank,'last_bind_evidence',{}) or {}).items()}
        report['speaker_binding_evidence']={'stage':'prepare','caption_assist_cue_ids':sorted(caption_binding_ids),
                                            'local_decisions':bind_decisions,'cues':{},'identity_gate':not line_voice_mode}
        report['speaker_bind_problems']=problems
        scoped=scope_bind_problems(problems,binding_cues) if not line_voice_mode else []
        issues.extend(scoped)
        unresolved={i['cue'] for i in scoped if i.get('kind')=='speaker_unresolved'}
        for cue in all_cues:
            local=cue.get('local_speaker') if line_voice_mode else cue.get('speaker')
            if cue['id'] in selected_ids:
                decision=bind_decisions.get(local)
                attempted=cue['id'] in binding_ids and local in binding_vectors
                reason=('preserved_short_policy' if cue['id'] in short_preserved else
                        'other_original_fallback' if cue['id'] not in binding_ids else
                        'no_local_speaker' if local is None else
                        'embedding_missing' if local not in binding_vectors else
                        decision.get('reason','decision_not_recorded') if decision else 'decision_not_recorded')
                report['speaker_binding_evidence']['cues'][cue['id']]={
                    'local_speaker':local,'binding_eligible':cue['id'] in binding_ids,'attempted':attempted,
                    'caption_assist':cue['id'] in caption_binding_ids,'reason':reason,
                    'decision':None if decision is None else dict(decision),'mapped_actor':mapping.get(local)}
            if line_voice_mode:
                own_floor=prefer_own_seconds(self.config)
                mode=_units.voice_mode(local,bind_decisions,merged_labels,cue_seconds=float(cue['end'])-float(cue['start']),
                    prefer_own_seconds=own_floor)
                ident=mapping.get(local)
                if mode=='bank' and ident is None:raise ValueError('Bank voice mode has no captured mapping')
                cue['speaker']=ident if mode=='bank' else 'line-'+cue['id']
                cue['voice_source']={'version':'voice-source-v1','mode':mode,'local_speaker':local,
                    'bank_ident':ident,'bind_reason':(bind_decisions.get(local) or {}).get('reason'),
                    'line_reason':None if mode=='bank' else 'merged_label' if local in merged_labels else
                                  'prefer_own_long_line' if float(cue['end'])-float(cue['start'])>=own_floor else
                                  (bind_decisions.get(local) or {}).get('reason','unresolved_local_speaker')}
                continue
            cue['local_speaker']=cue['speaker'];cue['speaker']=mapping.get(cue['speaker'])
            if cue['id'] in binding_ids and not cue['speaker'] and cue['id'] not in unresolved:
                issues.append({'kind':'speaker_unresolved','cue':cue['id']})
        caption_plan=None
        if caption_inputs is not None:
            # Source->target policy layer: raw cues stay untouched; only source-WORD uncertainty of usable targets is
            # released, and a target whose source is still kept original is refused with its issues restored.
            caption_plan=_ctf.plan_window(caption_inputs,job_id=job_id,revision=revision,source_identity=source_identity,
                all_cues=all_cues,speaker_evidence=speaker_evidence,issues=issues,origin=origin,end=end,
                render_start=render_start,render_end=render_end,caption_text=caption_text)
            _ctf.release_word_uncertainty(issues,caption_plan)
            report['caption_target_plan']=caption_plan
        if blocked():return save('needs_review')
        if caption_plan and _ctf.settle(caption_plan,kept_ids,issues) and blocked():return save('needs_review')
        # A clean solo source excerpt of this anonymous speaker is the reference. Its source transcript is OPTIONAL:
        # all-clear phrases supply it (reference+transcript cloning); otherwise it is '' (reference-only cloning).
        # The reference is attached to the bank only AFTER admission (program-cut source excerpt, B receipt,
        # provenance, no Sem root); a refused cut is removed and its cues keep original audio.
        def bank_reference_failed(ident,reason):
            if not line_voice_mode:
                issues.extend(reference_unavailable_issues(ident,active,reason));return
            report.setdefault('bank_reference_fallbacks',{})[ident]=reason
            for cue in all_cues:
                if cue.get('speaker')!=ident:continue
                cue['speaker']='line-'+cue['id']
                cue['voice_source'].update(mode='line-own',line_reason=reason)
        for ident in sorted({c['speaker'] for c in active
                             if not line_voice_mode or not c['speaker'].startswith('line-')}):
            if bank.voice(ident).get('reference'):
                report.setdefault('bank_reference_reuse',{})[ident]={
                    'reference_sha256':bank.voice(ident).get('reference_sha256'),
                    'reference':bank.voice(ident)['reference'],'basis':'existing-frozen-bank-reference'}
                continue
            reference=folder/(ident+'-reference.wav')
            try:
                choice=make_reference(speech,_ctf.reference_cues(caption_plan,[c for c in all_cues if c['speaker']==ident and c['id'] not in kept_ids]),
                                      origin,rate,reference,limit=self.config['actor_reference_seconds'],speech_regions=turns)
                # Freeze only source audio that the Vox renderer can actually use. A short/quiet/invalid candidate
                # stays unattached; future windows may try again. Existing accepted bank references are untouched.
                validate_reference_file(reference,limit=self.config['actor_reference_seconds'])
                if line_voice_mode and choice['seconds']<overlap_policy(self.config)['bank_min']:
                    raise ValueError('bank_reference_below_3_seconds')
            except ValueError as error:
                bank_reference_failed(ident,str(error));continue
            refused=_beta.admit_reference(self.config,self._admission_window,original,speech,active,all_cues,
                ident,bank.voice(ident),reference,choice,self.config.cache,report)
            if refused:
                bank_reference_failed(ident,refused);continue
            bank.attach_reference(ident,reference,choice['text'],source_intervals=choice['intervals'])
        if line_voice_mode:
            for cue in active:
                ident=cue['speaker']
                if not ident.startswith('line-'):continue
                reference=folder/(ident+'-reference.wav')
                try:
                    choice=make_line_reference(speech,cue,turns,origin,rate,reference,
                        limit=self.config['actor_reference_seconds'],
                        allow_extension=overlap_policy(self.config)['extend_merged'] or cue['id'] not in merged_cue_ids)
                    voice={'engine':'actor','actor_id':ident,'reference':str(reference.resolve()),
                           'reference_sha256':file_hash(reference),'reference_text':choice['text'],
                           'source_intervals':choice['intervals'],'line_reference':choice['line_reference']}
                    cue['line_reference']=dict(choice['line_reference'])
                    refused=_beta.admit_reference(self.config,self._admission_window,original,speech,active,all_cues,
                        ident,voice,reference,choice,self.config.cache,report)
                    if refused:raise ValueError(refused)
                    line_voices[ident]=voice
                except ValueError as error:
                    issues.append({'kind':'actor_reference_unavailable','speaker':ident,'cue':cue['id'],
                                   'reason':str(error),'provenance':LINE_REFERENCE_VERSION})
                    report.setdefault('line_reference_refusals',{})[cue['id']]=str(error)
            refused=report.get('line_reference_refusals') or {}
            if refused and overlap_policy(self.config)['borrow_reference']:
                def has_voice(other):
                    ident=other.get('speaker')
                    return isinstance(ident,str) and (ident in line_voices or
                        (not ident.startswith('line-') and bool(bank.voice(ident).get('reference'))))
                picks=choose_reference_donors(active,set(refused),has_voice,turns)
                borrowed={}
                for cue in active:
                    pick=picks.get(cue['id'])
                    if pick is None:continue
                    donor=pick['donor']
                    cue['voice_source'].update(mode='borrowed-reference',donor_cue_id=donor['id'],donor_speaker=donor['speaker'],
                        same_raw_label=pick['same_raw_label'],gap_seconds=pick['gap_seconds'],pitch_mismatch=pick['pitch_mismatch'],
                        refused_reason=refused[cue['id']])
                    cue['speaker']=donor['speaker']
                    borrowed[cue['id']]={'donor_cue_id':donor['id'],'donor_speaker':donor['speaker'],
                        'same_raw_label':pick['same_raw_label'],'gap_seconds':pick['gap_seconds'],'pitch_mismatch':pick['pitch_mismatch'],
                        'refused_reason':refused[cue['id']]}
                if borrowed:
                    issues[:]=[i for i in issues if not (i.get('kind')=='actor_reference_unavailable' and i.get('cue') in borrowed)]
                    report['line_reference_borrowed']=borrowed
        if blocked():return save('needs_review')
        stamp=time.perf_counter()
        translation_rows=[]
        for cue in active:
            owner.check()
            if _ctf.skip_source_path(caption_plan,cue):continue   # verified caption target supplies the English
            # Owner rule (2026-10-19): subtitles are for viewing only. They never supply or restyle the translation.
            if cue.get('unclear'):
                issues.append({'kind':'uncertain_transcript','cue':cue['id'],**({'reason':cue['nonverbal']} if cue.get('nonverbal') else {})})
            else:
                translation_rows.append({'id':cue['id'],'text':cue['text'],'language':cue.get('language') or language['language']})
        if translation_rows:
            request=folder/'translation-input.json';response=folder/'translation-output.json'
            atomic_json(request,translation_rows)
            save('translating')
            # One batched call through the current owner's language client (AnalysisService GPU child, cached there).
            # A cue the translator cannot translate keeps its original audio (cue-scoped issue); the rest of the window
            # is still dubbed. Whole-call problems (model cannot load, lease ended, cancel) still raise as before.
            owner.check()
            answers=self._language_client(owner).english_batch(
                [{'text':row['text'],'language':row['language']} for row in translation_rows])
            if len(answers)!=len(translation_rows):
                raise MovieConfigurationError('translation_batch_mismatch','Translator answered %d rows for %d'%(len(answers),len(translation_rows)))
            # A model that rambles past its token budget or answers nothing for an ordinary short line is asked again with a
            # word budget that fits the line's own length (Owner 2026-10-19: every spoken line is dubbed).
            answers=list(answers)
            retry=[i for i,answer in enumerate(answers)
                   if (isinstance(answer,MovieContentFallback) and getattr(answer,'fault_code',None) in TRANSLATION_RETRY_CODES)
                   or (isinstance(answer,tuple) and translation_artifact_reason(answer[0],translation_rows[i]['text']))]
            if retry:
                seconds={c['id']:float(c['end'])-float(c['start']) for c in active}
                second=self._language_client(owner).english_batch(
                    [{'text':translation_rows[i]['text'],'language':translation_rows[i]['language'],
                      'max_words':translation_retry_budget(seconds[translation_rows[i]['id']])} for i in retry])
                for i,answer in zip(retry,second):answers[i]=answer
            translated_rows=[]
            for row,answer in zip(translation_rows,answers):
                if isinstance(answer,MovieContentFallback):
                    issues.append({'kind':'uncertain_transcript','cue':row['id'],'reason':answer.fault_code})
                    continue
                if isinstance(answer,Exception):raise answer
                english,basis=answer
                artifact=translation_artifact_reason(english,row['text'])
                if artifact:
                    # The model answered with commentary, Cyrillic or the source spelled in Latin letters: not a translation.
                    issues.append({'kind':'uncertain_transcript','cue':row['id'],'reason':artifact})
                    continue
                if already_english(row['text'],english):
                    # The line is already in English (an English scene in a foreign film): it keeps its original audio.
                    issues.append({'kind':'uncertain_transcript','cue':row['id'],'reason':'already_english'})
                    continue
                translated_rows.append({'id':row['id'],'english':english,'translation':basis})
            atomic_json(response,translated_rows)
            translated={r['id']:r for r in json.loads(response.read_text(encoding='utf-8'))}
            for cue in active:
                if cue['id'] in translated:cue.update(translated[cue['id']])
        timing['translation']=time.perf_counter()-stamp
        # Translation is the last cue mutation, so this final policy is the one
        # persisted and later recomputed from the plan's own evidence.
        if blocked():return save('needs_review')
        policy=report['original_audio_policy']
        if persisted:
            # Final policy: a persisted keep of a selected cue must be reproduced exactly.
            late=[i for i in persisted_protection_issues(persisted['active'],cues,policy) if i not in issues]
            if late:
                issues.extend(late)
                if blocked():return save('needs_review')
                policy=report['original_audio_policy']
        _validate_policy(policy,origin=origin,end=end,synthesis_cues=active+ownership_accounted(report,all_cues),channel_layout=layout,
            beta_original_fallback=report.get('beta_original_fallback'),beta_enabled=_beta.beta_settings(self.config) is not None,
            channels=original.shape[1],all_cues=all_cues,speech_regions=turns,
            render_start=render_start,render_end=render_end)
        voices={ident:(line_voices[ident] if line_voice_mode and ident.startswith('line-') else bank.voice(ident))
                for ident in {c['speaker'] for c in active}}
        if line_voice_mode:
            # The captured bind's commit view covers every mapping, even if its
            # last cue later keeps original. Preserve exact bank dicts here;
            # render uses only the subset actually requested by synthesis cues.
            voices.update({ident:bank.voice(ident) for ident in mapping.values() if ident is not None})
        # This stricter plan is for the staged progressive renderer. Existing
        # canonical takes keep their original offline placement until re-review.
        timing_regions=[{'start':t['start'],'end':t['end']} for t in turns]
        timing_regions.extend({'start':c['start'],'end':c['end']} for c in all_cues if c['id'] not in selected_ids)
        # Kept original speech is reserved source audio, not free time.
        timing_regions.extend({'start':a,'end':b} for a,b in protected_spans(policy))
        timing_regions.extend({'start':a,'end':b} for a,b in _beta.fallback_spans(report.get('beta_original_fallback') or {},all_cues))
        timing_regions.extend({'start':a,'end':b} for a,b in _subs.timing_blocked(report))
        if persisted:timing_regions.extend({'start':r['start'],'end':r['end']} for r in persisted['active'])
        if ownership is not None:
            # Non-owned speech is never free time for this window's takes.
            timing_regions.extend({'start':a,'end':b} for item in ownership['items'] for a,b in item['speech_pieces'])
        report['progressive_timing']=plan_dubbing_timing(_ctf.render_view(caption_plan,active,all_cues)[0],origin=origin,end=end,speech_regions=timing_regions)
        report['preparation_seconds']=time.perf_counter()-started
        report['preparation_reused']=False
        fenced_json(store,fence,plan_path,{'key':plan_key,'engine_fingerprint':self.config.engine_fingerprint(),
                              'voices':voices,'cues':active,'all_cues':all_cues,
                              'original_audio_policy':policy,'channel_layout':layout,
                              'channel_layout_evidence':layout_evidence,
                              'protection_scope':protection.scope if protection else None,
                              'persisted_original_protection':persisted['active'] if persisted else [],
                              **({'first_pass':_fpf.record(first_pass)} if first_pass is not None else {'speech_sha256':file_hash(stem_path)}),
                              'pcm_sha256':pcm_identity,'report':report})
        if prepare_only:return save('prepared')
        return self._render_prepared(original,speech,rate,origin,end,active,all_cues,voices,
                                     folder,store,job_id,owner,revision,report,started,protection=protection,removal=removal,
                                     claimant=claimant,fence=fence)

    def _render_prepared(self,original,speech,rate,origin,end,cues,all_cues,voices,
                         folder,store,job_id,owner,revision,report,started,protection=None,removal=None,
                         claimant=None,fence=None):
        if report.get('beta_dialogue_ownership') is None:
            return self._render_attempt(original,speech,rate,origin,end,cues,all_cues,voices,
                                        folder,store,job_id,owner,revision,report,started,protection=protection,removal=removal)
        # v3 (caption_targets): one claimant (run id) holds this window's line claims across all its attempts.
        # A window whose atomic line-ledger commit lost a race to another window re-renders from a fresh snapshot
        # (the winner's commit is final; this run's claims and synthesized artifacts are kept and reused), and
        # fails closed after the limit. A take that is not audible in the rendered mix is kept original and the
        # window re-renders without it. Every unpublished claim is released on any exit.
        import copy
        import sqlite3
        pristine=copy.deepcopy((cues,all_cues,report));lost=[];inaudible=[];forced={};boosts={}
        if claimant is None:claimant=_lines.new_claimant(_lines.window_key(report['render_start'],report['render_end']))
        attempt=0;outcome=None
        try:
            while True:
                attempt+=1
                if attempt>1:cues,all_cues,report=copy.deepcopy(pristine)
                report['beta_line_ledger']={'attempt':attempt,'lost_races':copy.deepcopy(lost),'run_id':claimant['run_id'],
                                            'chunk_fence':fence,'not_audible_retries':copy.deepcopy(inaudible)}
                try:
                    return self._render_attempt(original,speech,rate,origin,end,cues,all_cues,voices,
                                                folder,store,job_id,owner,revision,report,started,protection=protection,
                                                removal=removal,claimant=claimant,fence=fence,forced=dict(forced),boosts=dict(boosts))
                except LineCommitRace as race:
                    lost.append({'attempt':attempt,'races':race.races})
                    if len(lost)>=LINE_COMMIT_ATTEMPTS:
                        outcome={'kind':'line_ledger_commit_race','attempts':len(lost)};break
                except InaudibleTakes as error:
                    inaudible.append({'attempt':attempt,'cues':error.cue_ids,'rows':error.rows})
                    rows={row.get('take_id'):row for row in error.rows if isinstance(row,dict)}
                    fresh={};lifted=False
                    for ident in error.cue_ids:
                        if ident in forced:continue
                        if (rows.get(ident) or {}).get('reason') in AUDIBLE_BOOST_REASONS and boosts.get(ident,0.)+AUDIBLE_BOOST_STEP_DB<=AUDIBLE_BOOST_MAX_DB:
                            boosts[ident]=boosts.get(ident,0.)+AUDIBLE_BOOST_STEP_DB;lifted=True
                        else:fresh[ident]='take_not_audible_in_render'
                    if not fresh and not lifted:
                        outcome={'kind':'take_not_audible_in_render','cues':error.cue_ids};break
                    forced.update(fresh)
        finally:
            try:_lines.release(store,job_id=job_id,revision=revision,claimant=claimant,line_ids=None,
                               reason='window_render_finished')
            except sqlite3.Error:pass        # unreleased claims expire with their fence, lease or dead holder
        cues,all_cues,report=copy.deepcopy(pristine)
        report['beta_line_ledger']={'attempt':attempt,'lost_races':lost,'not_audible_retries':inaudible,
                                    'run_id':claimant['run_id'],'status':'retries_exhausted'}
        report['issues'].append(outcome)
        report.update(state='needs_review',elapsed_seconds=time.perf_counter()-started)
        fenced_json(store,fence,folder/'actor-receipt.json',report)
        return report

    def _render_attempt(self,original,speech,rate,origin,end,cues,all_cues,voices,
                        folder,store,job_id,owner,revision,report,started,protection=None,removal=None,
                        claimant=None,fence=None,forced=None,boosts=None):
        import numpy as np
        import soundfile as sf
        issues=report['issues']
        voices={ident:voices[ident] for ident in {c['speaker'] for c in cues}}
        render_start,render_end=report['render_start'],report['render_end']
        lfe=3 if original.shape[1]==6 else None
        ledger=DialogueLedger(store,job_id,revision)
        canonical=ledger.between(origin,end)
        ownership=report.get('beta_dialogue_ownership');line_records={};line_state=None;claimed={}
        if ownership is not None:
            if fence is not None:_lines.check_fence(store,fence)       # a superseded worker stops before any work
            # Snapshot of what other windows committed: canonical takes (above) and window records.
            line_state=line_ledger_state(store,job_id,revision,report,origin,end,canonical)
            line_state['rendered']={}
        def save(state):
            report.update(state=state,elapsed_seconds=time.perf_counter()-started)
            from .resource_budget import ResourceBudget
            report.setdefault('memory_stages',[]).append({'stage':state,'ram_free_mib':ResourceBudget.ram_free()})
            fenced_json(store,fence,folder/'actor-receipt.json',report)
            return report
        layout=report.get('channel_layout')
        policy=report.get('original_audio_policy')
        if layout is None or policy is None:
            raise ValueError('Prepared actor report lacks its channel layout or original-audio policy')
        own=protected_spans(policy)
        # Persisted keeps are re-read now: a reused plan may predate another
        # window's decision. They join this window's walls and mix masks.
        persisted=protection.between(origin,end,window=(origin,end),original=original,rate=rate) if protection else None
        if persisted:
            kept=set(policy['preserved_cue_ids']);synthesis={c['id'] for c in cues}
            selected=[c for c in all_cues if c['id'] in kept or c['id'] in synthesis]
            for issue in persisted['issues']+persisted_protection_issues(persisted['active'],selected,policy):
                if issue not in issues:issues.append(issue)
            report['persisted_original_protection']={'version':persisted['version'],'active':persisted['active'],
                'stale_revision_rows':len(persisted['stale_revision']),'pcm_evidence':persisted['pcm_evidence']}
        protected=merge_spans(own,[(r['start'],r['end']) for r in persisted['active']] if persisted else [])
        # Recheck against the ledger as it is now; a reused plan may predate a
        # neighbouring window's committed take.
        for begin,stop in protected:
            for item in canonical:
                left=min(item['start'],item['replacement_start'],item['source_start'])
                right=max(item['end'],item['replacement_end'],item['source_end'])
                if min(stop+PROTECTION_TRANSITION_SECONDS,right)>max(begin-PROTECTION_TRANSITION_SECONDS,left):
                    issues.append({'kind':'kept_original_conflicts_with_canonical_take','canonical_id':item['id'],
                                   'start':begin,'end':stop})
        if issues:return save('needs_review')
        source_cues,source_all_cues=cues,all_cues
        cues,all_cues=_ctf.render_view(report.get('caption_target_plan'),cues,all_cues)   # target units; raw lists untouched
        cues,beta_spans=_beta.window_fallback(self.config,report,cues,all_cues)
        if beta_spans:protected=merge_spans(protected,beta_spans)
        cues,subs_walls=_subs.fallback_unprocessed(report,cues)
        if subs_walls:protected=merge_spans(protected,subs_walls)
        if ownership is not None:
            if forced:
                # v3: takes measured inaudible in an earlier attempt's rendered mix keep original this time.
                runtime=report.setdefault('beta_runtime_fallback',{})
                for ident,reason in sorted(forced.items()):runtime.setdefault(ident,[]).append(reason)
                cues,_,_,forced_spans=_beta._cascade(report,cues,list(cues),None,all_cues)
                if forced_spans:protected=merge_spans(protected,forced_spans)
            line_state['owned_unit_spans']=[[float(c['start']),float(c['end'])] for c in report['cues']]   # every owned cue
            def claim(lines):
                return _lines.claim_lines(store,job_id=job_id,revision=revision,claimant=claimant,lines=lines,
                                          same_line=_own.significant_overlap,fence=fence)
            cues,boundary_spans,line_records,claimed=boundary_render_plan(report,ownership,cues,all_cues,canonical,ledger,
                                                                          ownership.get('source_identity'),claim=claim)
            if boundary_spans:protected=merge_spans(protected,boundary_spans)
            # Claims of units that the boundary cascade moved to original are released at once (a same-line
            # claimant elsewhere may then synthesize the line).
            kept={c['id'] for c in cues}
            idle=[r['line_id'] for ident,r in claimed.items() if ident not in kept and r['status'] in SYNTHESIS_GRANTS]
            if idle:_lines.release(store,job_id=job_id,revision=revision,claimant=claimant,line_ids=idle,
                                   reason='unit_kept_original_at_render')
        save('cloning')
        # The actor runtime is acquired only for a cue that needs a new take;
        # all-kept, no-speech and fully canonical windows never load it.
        acquired=None
        def actor():
            nonlocal acquired
            if acquired is None:
                if _beta.beta_settings(self.config):
                    if self.renderer is not None:self.renderer.close()
                    self.renderer=_beta.make_actor_renderer(line_voice_config(self.config),owner,pipeline_context=_beta.admission_context(
                        self.config,self._admission_window,original,speech,source_cues,source_all_cues,voices,self.config.cache,report=report))
                elif self.renderer is None:self.renderer=ActorRenderer(self.config,owner)
                self.renderer.owner=owner
                self.renderer.prepare_references(voices.values())
                acquired=self.renderer
            return acquired
        def take_for(cue,seed,render,keep_on_lost=False):
            """v3: a claimed line is synthesized at most once per seed: a take recorded in its claim (earlier attempt,
            crashed or released run of this window) is reused when its file still has the recorded sha256. v4: the
            take is published from a private per-claim copy (beta_line_ledger.artifact_name), never from the
            renderer's shared file. If the
            claim was lost meanwhile the take is not recorded; keep_on_lost (verification retry) returns it anyway and
            the commit refuses it (claim_not_held), otherwise LineClaimLost keeps the cue original."""
            grant=claimed.get(cue['id'])
            if grant is None:return render()
            reused=take_reused(cue,seed)
            if reused is not None:return reused
            _lines.renew(store,job_id=job_id,revision=revision,claimant=claimant,fence=fence)
            path,receipt=render()
            return take_publish(cue,seed,path,receipt,keep_on_lost)
        def take_reused(cue,seed):
            """The claim's recorded take for this seed when its file still has the recorded sha256, else None."""
            grant=claimed.get(cue['id'])
            if grant is None:return None
            key='seed-%d'%seed
            artifact=(grant.get('artifacts') or {}).get(key)
            if artifact:
                path=Path(artifact['audio'])
                if path.is_file() and file_hash(path)==artifact['receipt'].get('sha256'):
                    grant.setdefault('reused_artifacts',[]).append(key)
                    published_copy[str(path)]=path
                    return path,dict(artifact['receipt'])
            return None
        def take_publish(cue,seed,path,receipt,keep_on_lost=False):
            """Publish a fresh take of a claimed line (private per-claim copy + record_artifact). Run ONLY by the window's
            coordinator thread, in canonical cue order (FIX 6: renders may run concurrently, commits never do)."""
            grant=claimed.get(cue['id'])
            if grant is None:return path,receipt
            key='seed-%d'%seed
            # v4 (W2 P2-3): the renderer's file name is shared by every holder of this line; publish only a private
            # copy under the per-claim name, written atomically and checked against the take's own sha256.
            claims_dir=Path(self.config.cache)/job_id/'actor-utterances'/'claims'
            claims_dir.mkdir(parents=True,exist_ok=True)
            private=claims_dir/_lines.artifact_name(grant['line_id'],grant['token'],claimant['run_id'],key,
                                                    Path(str(path)).suffix or '.wav')
            temp=private.with_name(private.name+'.%d.tmp'%os.getpid())
            temp.write_bytes(Path(path).read_bytes())
            if file_hash(temp)!=receipt.get('sha256'):
                temp.unlink()                                   # the shared file changed under us: never publish it
                if keep_on_lost:return path,receipt             # (verification retry: the commit refuses it)
                raise LineClaimLost(cue['id'])
            os.replace(temp,private)
            # Readback still reads the rendered file; the line switches to this private copy before the mix
            # (publish_private_copies), after re-checking that the rendered file was not changed meanwhile.
            published_copy[str(path)]=private
            artifact={'audio':str(private),'receipt':dict(receipt,**({'audio':str(private)} if 'audio' in receipt else {}))}
            if not _lines.record_artifact(store,job_id=job_id,revision=revision,claimant=claimant,line_id=grant['line_id'],
                                          token=grant['token'],key=key,artifact=artifact,fence=fence):
                if keep_on_lost:return path,receipt
                raise LineClaimLost(cue['id'])
            grant.setdefault('artifacts',{})[key]=artifact
            return path,receipt
        published_copy={}
        verified={}
        def readback_of(english,audio,seconds):
            """What an English take sounds like to the recognizer (no VAD trimming of short clips) and whether it changed a
            number or a negation. Shared by the seed search and the final verification, so both read the same result."""
            heard,_=self._speech(owner,str(audio),'en',vad=seconds>=2,beam_size=1)
            text=' '.join(x['text'] for x in heard)
            fast_error=word_error(english,text);beam=1
            if fast_error>.25:
                heard,_=self._speech(owner,str(audio),'en',vad=seconds>=2,beam_size=5)
                text=' '.join(x['text'] for x in heard);beam=5
            return {'readback':text,'readback_word_error':word_error(english,text),
                    'readback_fast_error':fast_error,'readback_beam_size':beam,
                    'critical_readback_issues':critical_readback_changes(english,text)}
        def concise_english(cue,window):
            """The same line, translated again within a word budget that fits `window` (None: no budget applies, or the
            translator could not)."""
            budget=concise_budget(cue['english'],window)
            code=(report.get('language') or {}).get('language')
            if budget is None or not code or not cue.get('text'):return None
            letters=sum(1 for ch in str(cue['text']) if ch.isalpha())
            if letters<=3 or (len(str(cue['text']).split())<=1 and float(cue.get('probability') or 0.)<.9):
                return None       # a sliver or a doubtful single word: shortening what may be a misheard fragment invents speech
            answer=self._language_client(owner).english_batch([{'text':cue['text'],'language':code,'max_words':budget}])[0]
            if isinstance(answer,Exception):
                if _beta.is_cue_unavailable(self.config,answer) or isinstance(answer,MovieContentFallback):return None
                raise answer
            english=str(answer[0]).strip()
            return english if english and english!=cue['english'] else None
        def take_check(cue,path,receipt):
            if receipt['sha256'] not in verified:
                verified[receipt['sha256']]=readback_of(receipt.get('requested_text') or cue['english'],path,receipt['seconds'])
            entry=verified[receipt['sha256']]
            return {'critical':bool(entry['critical_readback_issues']),'error':entry['readback_word_error']}
        def fit_take(cue,path,receipt,window,again=None):
            """Fit the take into its slot; when it still overruns or does not read back clean and `again(seed)` can render
            the same words on another seed, search the retry seeds (choose_take). The result is published under the cue's
            seed-42 claim key."""
            policy=overlap_policy(self.config)
            def fit(p,r):
                fitted=fit_rendered_take(p,r,window,ffmpeg=self.config['ffmpeg'],runner=owner.run,max_speed=policy['fit_max_speed'])
                return deess_rendered_take(*fitted) if policy['deess_takes'] else fitted
            first=fit(path,receipt)
            if again is None:return first
            return choose_take(*first,window,render=again,fit=fit,check=lambda p,r:take_check(cue,p,r),
                               recoverable=lambda error:_beta.is_cue_unavailable(self.config,error),
                               concise=lambda:concise_english(cue,window) if overlap_policy(self.config)['concise_retranslation'] else None)
        def vox_concurrency_configured():
            """FIX 6: the upper bound of lines in flight. actor_beta.vox.max_instances: "auto" (default) -> the cap
            max_instances_cap (default 2; the window then runs 1 until the measured admission allows more); an explicit
            integer overrides (1 = the sequential loop below, unchanged). Not on the actor_beta path: 1."""
            beta=_beta.beta_settings(self.config)
            if not beta:return 1
            vox=beta.get('vox') or {}
            value=vox.get('max_instances',vox.get('instances','auto'))
            try:
                if str(value).strip().lower()=='auto':return max(1,int(vox.get('max_instances_cap') or 2))
                return max(1,int(value))
            except (TypeError,ValueError):return 1
        def render_window_seconds(cue):
            next_start=min([c['start'] for c in all_cues if c['start']>cue['start']]+
                [a-PROTECTION_TRANSITION_SECONDS for a,_ in protected if a>cue['start']],default=end)
            budget=max(0.,next_start-cue['start']-.04)
            if _beta.beta_settings(self.config):
                meta=_units.silence_slack(cue,raw_turns=report.get('raw_turns') or (),
                    transcript=report.get('transcript') or (),all_cues=all_cues,protected=protected,
                    placements=[dict(item,cue_id=item.get('cue',item.get('id'))) for item in canonical],
                    origin=origin,analysis_end=end,window_limit=line_state['new_take_end_limit'] if line_state else end)
                report.setdefault('render_silence_slack',{})[cue['id']]=meta
                budget=min(budget,max(0.,meta['allowed_end']-cue['start']))
            return budget

        def render_concurrently():
            """FIX 6 (bounded Vox concurrency). Three steps, one coordinator:
            1. plan, in canonical cue order, exactly as the sequential loop: window, carried/own prior take;
            2. render: up to `bound` takes in flight through the window renderer's submit() (the engine host orders them
               earliest-deadline-first, then by priority = canonical order; one RenderJob per line and seed, so no line
               is synthesized twice; a claim's recorded take is reused without rendering);
               bound = min(actor_beta.vox.max_instances, the measured admission: the GPU lease's granted budget over
               the measured per-instance peak; 1 until measured) via renderer.admitted_concurrency();
            3. commit: THIS thread consumes results in canonical order and alone publishes the private per-claim copies
               (take_publish -> record_artifact) and appends the lines, so identity, timing and publication order are
               those of the sequential loop. Takes are immutable per claim (content-addressed renderer file + private
               per-claim copy). Any abort (Sem preemption -> ResourceYield/InterruptedError, a window error) cancels
               every outstanding job (bounded: the child stops within its hard-cancel window) before it propagates.
            One renderer, one guard and one GPU lease (the job keeper) serve every line: no nested lease."""
            directory=self.config.cache/job_id/'actor-utterances'
            plan=[]
            for index,cue in enumerate(cues):
                owner.check()
                window=render_window_seconds(cue)
                if ownership is None:prior=ledger.match(cue,canonical)
                else:
                    prior=_own.own_prior(cue,canonical,window=[render_start,render_end],match=ledger.match,
                                         line_ids={k:v['line_id'] for k,v in line_records.items()})
                    if prior is not None:cue['english']=prior['text']
                if prior:
                    plan.append(('prior',cue,{**prior,'cue':cue['id'],'canonical_id':prior['id']}))
                    continue
                plan.append(('take',cue,window))
            takes=[i for i,entry in enumerate(plan) if entry[0]=='take']
            renderer=actor() if takes else None
            admitted=getattr(renderer,'admitted_concurrency',None)
            bound=1
            if takes and callable(admitted) and callable(getattr(renderer,'submit',None)):
                bound=max(1,min(vox_concurrency_configured(),int(admitted() or 1)))
            report['vox_concurrency']={'configured':vox_concurrency_configured(),'bound':bound,'takes':len(takes)}
            started={}                                          # plan index -> ('done', take) | ('job', RenderJob)
            waiting=list(takes)
            def submit_next():
                while waiting and sum(1 for kind,_ in started.values() if kind=='job')<bound:
                    i=waiting.pop(0)
                    _,cue,window=plan[i]
                    reused=take_reused(cue,42)
                    if reused is not None:
                        started[i]=('done',reused)
                        continue
                    if claimed.get(cue['id']) is not None:
                        _lines.renew(store,job_id=job_id,revision=revision,claimant=claimant,fence=fence)
                    started[i]=('job',renderer.submit(cue['english'],voices[cue['speaker']],directory,window,
                                                      context_key={'cue_id':cue['id']},job_id='cue-%s-s42'%cue['id'],
                                                      priority=-i,**vox_options(self.config)))
            try:
                if bound>1:
                    submit_next()
                for i,entry in enumerate(plan):
                    owner.check()
                    if entry[0]=='prior':
                        lines.append(entry[2])
                        continue
                    _,cue,window=entry
                    try:
                        if bound<=1:                            # admission allows one: the sequential take, unchanged
                            path,receipt=take_for(cue,42,lambda:fit_take(cue,*renderer.render(cue['english'],voices[cue['speaker']],
                                directory,window,context_key={'cue_id':cue['id']},**vox_options(self.config)),window,
                                again=lambda seed,text=None:renderer.render(text or cue['english'],voices[cue['speaker']],directory,window,
                                    seed=seed,context_key={'cue_id':cue['id']},**vox_options(self.config))))
                        else:
                            kind,value=started.pop(i)
                            if kind=='done':
                                path,receipt=value
                            else:
                                path,receipt=take_publish(cue,42,*fit_take(cue,*value.result(),window,
                                    again=lambda seed,text=None:renderer.render(text or cue['english'],voices[cue['speaker']],directory,window,
                                        seed=seed,context_key={'cue_id':cue['id']},**vox_options(self.config))))
                    except LineClaimLost:
                        report['beta_runtime_fallback'].setdefault(cue['id'],[]).append('line_claim_lost')
                        continue
                    except Exception as error:
                        if not _beta.is_cue_unavailable(self.config,error):raise
                        report['beta_runtime_fallback'].setdefault(cue['id'],[]).append('actor_unavailable:'+str(error))
                        continue
                    finally:
                        if bound>1:submit_next()
                    adopt_spoken_text(cue,receipt)
                    lines.append({'cue':cue['id'],'audio':str(path),**receipt,**dialogue_line_fields(cue),
                                  **(line_records.get(cue['id']) or {})})
            except BaseException as error:
                waiting.clear()
                if isinstance(error,(ResourceYield,InterruptedError)) and hasattr(renderer,'note_refusal'):
                    renderer.note_refusal('window_resource_wait:%s'%error)   # next windows fall back to 1
                for kind,value in list(started.values()):
                    if kind=='job':value.cancel('window_aborted')
                for kind,value in list(started.values()):
                    if kind=='job':value.wait(2.0)              # no take is written after the window gave up
                raise
        concurrent=vox_concurrency_configured()>1
        generated=np.zeros_like(original);intervals=[];lines=[]
        render_started=time.perf_counter()
        try:
            if concurrent:render_concurrently()
            for index,cue in enumerate(() if concurrent else cues):
                owner.check()
                # The take budget also stops before kept original speech.
                window=render_window_seconds(cue)
                if ownership is None:prior=ledger.match(cue,canonical)
                else:
                    # v3: only this window's own take of the line is reused; another window's take is carried.
                    prior=_own.own_prior(cue,canonical,window=[render_start,render_end],match=ledger.match,
                                         line_ids={k:v['line_id'] for k,v in line_records.items()})
                    if prior is not None:cue['english']=prior['text']
                if prior:
                    lines.append({**prior,'cue':cue['id'],'canonical_id':prior['id']})
                    continue
                try:
                    path,receipt=take_for(cue,42,lambda:fit_take(cue,*actor().render(cue['english'],voices[cue['speaker']],
                        self.config.cache/job_id/'actor-utterances',window,**({'context_key':{'cue_id':cue['id']}} if _beta.beta_settings(self.config) else {}),**vox_options(self.config)),window,
                        again=lambda seed,text=None:actor().render(text or cue['english'],voices[cue['speaker']],
                            self.config.cache/job_id/'actor-utterances',window,seed=seed,**({'context_key':{'cue_id':cue['id']}} if _beta.beta_settings(self.config) else {}),**vox_options(self.config))))
                except LineClaimLost:
                    report['beta_runtime_fallback'].setdefault(cue['id'],[]).append('line_claim_lost')
                    continue
                except Exception as error:
                    if not _beta.is_cue_unavailable(self.config,error):raise
                    report['beta_runtime_fallback'].setdefault(cue['id'],[]).append('actor_unavailable:'+str(error))
                    continue
                adopt_spoken_text(cue,receipt)
                lines.append({'cue':cue['id'],'audio':str(path),**receipt,**dialogue_line_fields(cue),
                              **(line_records.get(cue['id']) or {})})
        except (ResourceYield,InterruptedError) as error:
            if not _beta.is_resource_wait(error):raise        # explicit fatal/cancelled fault metadata wins
            report['resource_wait']=str(error)
            report['resource_fault']=_beta.fault_of(error)
            report['utterances']=lines
            return save('waiting_resources')
        finally:
            # Retain CPU models, release only our actor GPU session between
            # windows so a newly active Sem call can claim memory immediately.
            if acquired is not None:acquired.close()
        report['utterances']=lines
        cues,lines,beta_spans=_beta.after_takes(self.config,report,cues,lines,all_cues)
        if beta_spans:protected=merge_spans(protected,beta_spans)
        report['timing']['cloning']=time.perf_counter()-render_started
        # Read back complete short clips without VAD trimming away their first
        # syllable. Ordinary word error is diagnostic; critical meaning changes
        # still keep the cue original. Do not regenerate speech from ASR disagreement.
        verification_started=time.perf_counter()
        for cue,line in zip(cues,lines):
            owner.check()
            line['readback_policy']='ordinary-wer-diagnostic-critical-meaning-veto'
            if line.get('canonical_id') and line.get('readback') and line.get('readback_word_error',1)<=.25 and not critical_readback_changes(cue['english'],line['readback']):
                line['readback_ordinary_mismatch']=False
                continue
            if line['sha256'] not in verified:
                verified[line['sha256']]=readback_of(cue['english'],line['audio'],line['seconds'])
            line.update(verified[line['sha256']])
            line['readback_ordinary_mismatch']=line['readback_word_error']>.25
            if line.get('critical_readback_issues'):
                issues.append({'kind':'critical_readback_mismatch','cue':cue['id'],'changes':line['critical_readback_issues']})
            elif (overlap_policy(self.config)['reject_unintelligible'] and not line.get('canonical_id')
                  and line['readback_word_error']>=UNINTELLIGIBLE_ERROR
                  and text_similarity(cue['english'],line.get('readback') or '')<UNINTELLIGIBLE_SIMILARITY):
                # After every seed the take still says other words altogether: the original line plays instead of nonsense.
                issues.append({'kind':'critical_readback_mismatch','cue':cue['id'],'changes':['unintelligible']})
        report['timing']['verification']=time.perf_counter()-verification_started
        if claimed:
            # v4 (W2 P2-3): every new take is mixed and published from its private per-claim copy. A rendered file
            # whose bytes changed after the copy (a late write by another holder) invalidates its readback: the cue
            # keeps original audio.
            changed=[]
            for cue,line in zip(cues,lines):
                if line.get('canonical_id'):continue
                private=published_copy.get(str(line['audio']))
                if private is None:changed.append((cue['id'],'take_not_claimed'));continue
                if file_hash(Path(line['audio']))!=line.get('sha256'):changed.append((cue['id'],'take_file_changed'));continue
                line['audio']=str(private)
            if changed:
                for ident,reason in changed:report['beta_runtime_fallback'].setdefault(ident,[]).append(reason)
                cues,lines,_,beta_spans=_beta._cascade(report,cues,lines,None,all_cues)
                if beta_spans:protected=merge_spans(protected,beta_spans)
        mixing_started=time.perf_counter()
        reused={line['canonical_id'] for line in lines if line.get('canonical_id')}
        carry=[item for item in canonical if item['id'] not in reused and
               max(item['end'],item['replacement_end'])>render_start and
               min(item['start'],item['replacement_start'])<render_end]
        # v3 placement floor: an owner never places a take before its own render start, and a new take never ends
        # inside a committed window that rendered without it (line_state). Without ownership: origin and end.
        low,schedule=(origin,schedule_around_protection) if line_state is None else (
            render_start,schedule_within_limit(line_state['new_take_end_limit']))
        floor=max([low]+[item['end']+.04 for item in carry if cues and item['source_start']<cues[0]['start']])
        if _beta.beta_settings(self.config):
            from . import actor_beta_timing as _beta_timing
            schedule=schedule_with_silence_slack(schedule,report=report,all_cues=all_cues,analysis_origin=origin,
                analysis_end=end,window_limit=line_state['new_take_end_limit'] if line_state else end,canonical=canonical)
            cues,lines,placements,timing_issues,protected=_beta_timing.schedule_with_reschedule(
                issues,report,cues,lines,protected,end=end,schedule=schedule,merge=merge_spans,
                floor_for=lambda cs:max([low]+[item['end']+.04 for item in carry if cs and item['source_start']<cs[0]['start']]))
        else:
            placements,timing_issues=schedule(cues,[line['seconds'] for line in lines],
                                              origin=floor,end=end,protected=protected)
        for index,line in enumerate(lines):
            if line.get('canonical_id'):
                placements[index]={k:line[k] for k in ('start','end','source_start','source_end','early_seconds','speed','timing_overrun')}
                if line.get('timing_slack') is not None:placements[index]['timing_slack']=line['timing_slack']
                timing_issues=[i for i in timing_issues if i['cue']!=cues[index]['id']]
        occupied=sorted([dict(p,id=c['id']) for c,p in zip(cues,placements)]+carry,key=lambda x:x['start'])
        for left,right in zip(occupied,occupied[1:]):
            if left['end']>right['start']+.0001:
                timing_issues.append({'kind':'canonical_speech_collision','cues':[left['id'],right['id']]})
        issues.extend(timing_issues)
        report['placements']=placements
        cues,lines,placements,beta_spans=_beta.after_schedule(self.config,issues,report,cues,lines,placements,all_cues)
        if beta_spans:protected=merge_spans(protected,beta_spans)
        if line_state is not None:
            cues,lines,placements,beta_spans=line_audibility_gate(report,cues,lines,placements,all_cues,line_state)
            if beta_spans:protected=merge_spans(protected,beta_spans)
        if issues:return save('needs_review')
        pending=[]
        for cue,line,placement in zip(cues,lines,placements):
                if line.get('canonical_id'):
                    add_utterance(generated,line,origin,rate)
                    intervals.extend(take_removal_intervals(speech if removal is None else removal,line,origin,rate,lfe))
                    continue
                path=line['audio']
                voice,voice_rate=sf.read(path,dtype='float32')
                if voice_rate!=rate:raise ValueError('Actor output rate changed')
                left=round((placement['start']-origin)*rate)
                try:
                    pan,gain,levels=_ctf.mix_levels(actor_mix_levels,speech,voice,cue,origin,rate,lfe)
                except ValueError as error:
                    issues.append({'kind':'actor_gain_unresolved','cue':cue['id'],'reason':str(error)});continue
                lift=float((boosts or {}).get(cue['id'],0.))
                if lift:gain=gain*10**(lift/20);levels=dict(levels,audibility_boost_db=lift)
                line['mix_levels']=levels
                if left<0 or left+len(voice)>len(generated):
                    issues.append({'kind':'speech_crosses_window','cue':cue['id']});continue
                if protected:
                    _,allowed=original_protection_masks(len(voice),rate,(round(origin*rate)+left)/rate,protected,
                        transition_seconds=PROTECTION_TRANSITION_SECONDS)
                    if (allowed<1).any():
                        # Never truncate a take into kept original speech.
                        issues.append({'kind':'generated_speech_touches_protected_original','cue':cue['id'],
                                       'requires_rescheduling':True});continue
                generated[left:left+len(voice)]+=voice[:,None]*pan[None,:]*gain
                intervals.extend(take_removal_intervals(speech if removal is None else removal,cue,origin,rate,lfe))
                pending.append((cue,line,placement,pan,gain))
        for item in carry:
            add_utterance(generated,item,origin,rate)
            intervals.extend(take_removal_intervals(speech if removal is None else removal,item,origin,rate,lfe))
        report['carried_utterances']=[item['id'] for item in carry]
        pending,beta_spans=_beta.after_mix_takes(self.config,issues,report,pending,all_cues)
        if beta_spans:protected=merge_spans(protected,beta_spans)
        if issues:return save('needs_review')
        # Activity intervals are already tight (10 ms hysteresis edges); fade them over 10 ms, not the 40 ms of padded spans.
        envelope=replacement_envelope(len(original),rate,origin,merge_intervals(intervals),fade_seconds=.01)
        # With no protected span this equals replace_dialogue byte for byte.
        mixed,mix=protect_dialogue_mix(original,speech if removal is None else removal,generated,envelope,origin=origin,rate=rate,
            channel_layout=layout,protected_intervals=protected,transition_seconds=PROTECTION_TRANSITION_SECONDS)
        report['mix']=mix
        report['removal_policy']=SPEECH_ONLY_REMOVAL_VERSION
        if mix['requires_rescheduling']:
            # Carried or reused canonical takes are checked here too.
            issues.append({'kind':'generated_speech_touches_protected_original',
                           'generated_suppressed_core_frames':mix['generated_suppressed_core_frames'],
                           'generated_modified_transition_frames':mix['generated_modified_transition_frames'],
                           'requires_rescheduling':True})
        if mix['overrange_samples']:
            issues.append({'kind':'soundtrack_peak_overrange','samples':mix['overrange_samples']})
        if issues:return save('needs_review')
        if line_state is not None:
            # v3 published == heard: measure every new, reused and carried take in the rendered mix (mixed minus the
            # same mix without generated speech). A new or reused take that is not audible is never committed.
            bed,_=_bed_mix(original,speech if removal is None else removal,np.zeros_like(original),envelope,
                origin=origin,rate=rate,channel_layout=layout,protected_intervals=protected,
                transition_seconds=PROTECTION_TRANSITION_SECONDS)
            takes=[]
            for cue,line,placement,pan,gain in pending:
                voice,_=sf.read(line['audio'],dtype='float32')
                takes.append({'take_id':cue['id'],'kind':'generated','start':placement['start'],'end':placement['end'],
                              'samples':voice,'left':round((placement['start']-origin)*rate),'pan':pan,'gain':gain,'gating':True})
            def canonical_take(item,kind,ident,gating):
                voice,_=sf.read(item['audio'],dtype='float32')
                return {'take_id':ident,'kind':kind,'start':item['start'],'end':item['end'],'samples':voice,
                        'left':round(item['start']*rate)-round(origin*rate),'pan':item['channel_weights'],
                        'gain':item['voice_gain'],'gating':gating}
            for cue,line in zip(cues,lines):
                if line.get('canonical_id'):takes.append(canonical_take(line,'canonical_reused',cue['id'],True))
            for item in carry:takes.append(canonical_take(item,'carried',item['id'],False))
            line_state['rendered']=rendered_audibility(mixed,bed,origin,rate,render_start,render_end,takes)
            del bed
            silent=[t['take_id'] for t in takes if t['gating'] and not line_state['rendered'][t['take_id']]['audible']]
            if silent:raise InaudibleTakes(silent,[line_state['rendered'][i] for i in silent])
        owner.check()
        output=folder/'actor-English.wav'
        if line_state is not None:
            # v3: protection rows, new takes, line claims and this window's record commit in ONE IMMEDIATE
            # transaction, after the race checks against other windows' commits (beta_line_ledger).
            problems,races=_lines.commit(store,job_id=job_id,revision=revision,
                window=line_window_record(report,line_state,origin,end,lines,carry,pending),
                snapshot_ids=line_state['snapshot_canonical_ids'],utterances=[ledger.entry(*item) for item in pending],
                line_claims=line_claims(pending,line_state,claimed),protected=boundary_claims(ownership),
                same_line=_own.significant_overlap,protection=protection,claimant=claimant,fence=fence,
                protection_rows=protection_rows(policy,original,origin,rate,window_end=end,render_start=render_start,
                                                render_end=render_end,layout=layout) if protection else ())
            if races:raise LineCommitRace(races)
            if problems:
                issues.extend(problems)
                return save('needs_review')
            owner.check()
            _lines.fenced_write(store,fence,lambda:write_layout_wav(output,mixed,rate,layout))
        elif protection:
            # Kept decisions and new takes commit atomically before any output
            # exists; a conflicting decision committed meanwhile blocks here.
            problems=protection.commit(protection_rows(policy,original,origin,rate,window_end=end,
                render_start=render_start,render_end=render_end,layout=layout),
                [ledger.entry(*item) for item in pending])
            if problems:
                issues.extend(problems)
                return save('needs_review')
            owner.check()
            write_layout_wav(output,mixed,rate,layout)
        else:
            write_layout_wav(output,mixed,rate,layout)
            for cue,line,placement,pan,gain in pending:ledger.put(cue,line,placement,pan,gain)
        report.update(audio=str(output),sha256=file_hash(output),mix=mix,
                      state='candidate',quality_scope='Machine construction only; residual dialogue and soundtrack listening pending',
                      original_audio_protection={'version':policy['version'],'channel_layout':layout,
                          'preserved_cue_ids':policy['preserved_cue_ids'],
                          'protected_intervals':[list(i) for i in own],
                          'applied_protected_intervals':[list(i) for i in protected],
                          'covered_issues':policy['covered_issues'],
                          'cross_window_persistence':bool(protection),
                          'protection_scope':protection.scope if protection else None,
                          'human_listening_acceptance':False})
        report['timing']['scheduling_and_mix']=time.perf_counter()-mixing_started
        if _beta.beta_settings(self.config):
            from . import actor_beta_timing as _beta_timing
            _beta_timing.stamp_published(report,pending,cues,lines,placements,carry=carry,window_frames=len(original))
            _ctf.annotate_publication(report,source_all_cues)
            if line_state is not None:annotate_line_audibility(report,line_state)
        return save('candidate')

    def inspect_overlaps(self,recognition,origin,issues,language,vectors,folder,owner):
        """Inspect overlapping vocal candidates without silently replacing them."""
        import numpy as np
        import soundfile as sf
        from .overlap import OverlapSeparator
        from .resource_budget import ResourceBudget
        if ResourceBudget.ram_free() is not None and ResourceBudget.ram_free()<16384:
            return [{'state':'deferred_resources','reason':'Preserving CPU analysis memory reserve'}]
        ranges=[]
        for issue in issues:
            if issue['kind']!='overlapping_dialogue' or 'start' not in issue:continue
            begin=max(origin,issue['start']-.35);end=min(origin+len(recognition)/16000,issue['end']+.35)
            if end-begin<.5:continue
            if any(min(end,b)-max(begin,a)>0 for a,b in ranges):continue
            ranges.append((begin,min(end,begin+6)))
        if not ranges:return []
        if self.overlap_separator is None:
            self.overlap_separator=OverlapSeparator(self.config['overlap_checkpoint'],self.config['overlap_code_root'],
                dependency_root=self.config['overlap_dependency_root'] or None,threads=min(4,int(self.config['cpu_threads'])))
        records=[]
        for index,(begin,end) in enumerate(ranges[:3]):
            owner.check()
            samples=recognition[round((begin-origin)*16000):round((end-origin)*16000)]
            stems,metrics=self.overlap_separator.separate(samples,cancelled=owner.cancelled.is_set)
            record={'start':begin,'end':end,'metrics':metrics,'stems':[],
                    'state':'unverified_vocal_candidates','original_soundtrack_changed':False}
            for number,stem in enumerate(stems):
                owner.check();path=folder/f'overlap-{index}-{number}.wav'
                sf.write(path,stem,16000,subtype='FLOAT')
                transcript,_=self._speech(owner,path,language['language'] if language['probability']>=.65 else None)
                scores=[]
                if np.max(np.abs(stem))>1e-5:
                    embedding=self._embedding(owner,np.asarray(stem,dtype=np.float32))
                    if embedding is not None:
                        scores=sorted(((cosine(embedding,v),k) for k,v in vectors.items()),reverse=True)[:3]
                record['stems'].append({'audio':str(path),'transcript':transcript,'speaker_candidates':scores})
            records.append(record)
        return records
