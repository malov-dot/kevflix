"""Choose real solo dialogue for stable original-actor references."""
import math
from pathlib import Path

MIN_REFERENCE_SECONDS = 1.0
MIN_REFERENCE_RMS = 1e-4


def validate_reference_audio(audio, rate, *, limit):
    """Require the same usable source audio as VoxActorRenderer before a reference can be frozen.

    This validates only the untouched excerpt. It does not normalize, pad or otherwise change source PCM, and
    provenance/speaker admission still follows separately. A rejected candidate remains unattached so a later
    window can supply a usable reference for the same actor.
    """
    import numpy as np
    samples = np.asarray(audio)
    if samples.ndim == 2 and samples.shape[1] == 1:
        samples = samples[:, 0]
    if samples.ndim != 1 or not samples.size:
        raise ValueError('Actor reference must be non-empty mono audio')
    if not np.isfinite(rate) or float(rate) <= 0:
        raise ValueError('Actor reference sample rate is invalid')
    seconds = len(samples) / float(rate)
    if seconds < MIN_REFERENCE_SECONDS or seconds > float(limit):
        raise ValueError('Actor reference duration %.3fs outside %g..%gs' %
                         (seconds, MIN_REFERENCE_SECONDS, float(limit)))
    if not np.all(np.isfinite(samples)):
        raise ValueError('Actor reference contains non-finite audio')
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    if rms <= MIN_REFERENCE_RMS:
        raise ValueError('Actor reference RMS %.6g must exceed %.6g' % (rms, MIN_REFERENCE_RMS))
    return {'seconds': seconds, 'rms': rms, 'rate': int(rate), 'channels': 1}


def validate_reference_file(path, *, limit):
    """Validate the actual candidate WAV before provenance admission and attachment to the actor bank."""
    import soundfile as sf
    audio, rate = sf.read(str(path), dtype='float32', always_2d=True)
    if sf.info(str(path)).format != 'WAV':
        raise ValueError('Actor reference must be a WAV file')
    return validate_reference_audio(audio, rate, limit=limit)


def excerpt_solo_label(speech_regions,span,core):
    """The one raw speaker label of an excerpt, else None.

    ``span`` is the excerpt's ACTUAL frame-rounded extent (absolute seconds, padding included: the exact geometry
    make_reference cuts); ``core`` is the cue's speech_start/speech_end. Admitted only when every raw diarization
    turn intersecting the whole span carries the same non-empty label, and that label has real speech inside the
    core. Unknown-label speech, or speech of a second label anywhere in the span (including the padding and
    sequential speakers inside one ASR phrase), refuses the candidate. Nothing is trimmed or inferred."""
    s0,e0=span
    hits=[t for t in speech_regions if min(float(t['end']),e0)-max(float(t['start']),s0)>0]
    labels={t.get('speaker') for t in hits}
    if len(labels)!=1:return None
    label=next(iter(labels))
    if label is None or (isinstance(label,str) and not label.strip()):return None
    c0,c1=core
    if not any(min(float(t['end']),c1)-max(float(t['start']),c0)>0 for t in hits):return None
    return label


def _measured_db(loudness_db,cue,excerpt):
    """A finite dB for this candidate from the supplied mapping {cue id: dB} or callable(cue, excerpt) -> dB, else None."""
    try:
        value=loudness_db(cue,excerpt) if callable(loudness_db) else loudness_db.get(cue.get('id'))
        value=float(value)
    except (TypeError,ValueError,OverflowError,AttributeError):
        return None
    return value if math.isfinite(value) else None


def make_reference(speech,cues,origin,rate,path,*,limit=8,speech_regions=None,loudness_target_db=None,loudness_db=None):
    """``speech_regions``: the window's raw diarization turns (absolute seconds). Every candidate cue, selected or
    context-only, is checked against them over its actual excerpt span (excerpt_solo_label); a refused candidate is
    skipped and the next automatic candidate is tried (none left: no reference, original audio). Without that
    evidence no reference is cut (no evidence is synthesized). A cue already marked overlapped or turn_overlap=True
    is excluded as well. The excerpt geometry (0.06 s padding, 0.12 s joins) is unchanged.

    ``loudness_target_db`` / ``loudness_db`` (both None by default = the exact previous behaviour and return value):
    loudness-aware ordering. ``loudness_target_db`` is the active-frame dB of the line being cloned and
    ``loudness_db`` is {cue id: active-frame dB of that cue's dialogue-stem span} or callable(cue, excerpt) -> dB.
    Eligibility, geometry, stopping rule, time-ordered join and the untouched-PCM excerpts are unchanged (so the
    reference gate still sees exact stem slices); only the ORDER candidates are taken in changes: clear transcripts
    first (as before), then smallest |dB - target|, then longer, then earlier. A candidate with no finite measurement
    sorts after every measured one. The result gains a ``loudness`` receipt; no gain is ever applied to the audio."""
    import numpy as np
    import soundfile as sf
    if speech_regions is None:raise ValueError('No verified speaker evidence for actor reference cues')
    by_loudness=loudness_target_db is not None
    if by_loudness:
        try:target_db=float(loudness_target_db)
        except (TypeError,ValueError,OverflowError):target_db=math.nan
        if not math.isfinite(target_db) or loudness_db is None or not (callable(loudness_db) or hasattr(loudness_db,'get')):
            raise ValueError('Loudness-aware reference needs a finite target and per-cue loudness')
    elif loudness_db is not None:raise ValueError('loudness_db requires loudness_target_db')
    measured={}
    candidates=[]
    for cue in cues:
        if cue.get('overlapped') or cue.get('turn_overlap') is True:continue
        begin=cue.get('speech_start',cue['start'])
        end=cue.get('speech_end',cue['end'])
        if not .25<=end-begin<=limit:continue
        a=max(0,round((begin-origin-.06)*rate));b=min(len(speech),round((end-origin+.06)*rate))
        if b<=a:continue
        if excerpt_solo_label(speech_regions,(origin+a/rate,origin+b/rate),(begin,end)) is None:continue
        excerpt=speech[a:b]
        if not np.all(np.isfinite(excerpt)):continue
        # A quiet or invalid early cue must not displace a later usable source excerpt. The final joined mono
        # reference is validated again, including the renderer's full-duration RMS and minimum length.
        if float(np.max(np.sqrt(np.mean(np.square(excerpt,dtype=np.float64),axis=0))))<=MIN_REFERENCE_RMS:continue
        candidates.append((not cue.get('unclear',False),end-begin,cue,a,b))
        if by_loudness:measured[id(cue)]=_measured_db(loudness_db,cue,excerpt)
    if by_loudness:
        def delta(candidate):
            value=measured.get(id(candidate[2]))
            return math.inf if value is None else round(abs(value-target_db),6)
        candidates.sort(key=lambda c:(not c[0],delta(c),-round(c[1],6),c[3],c[4]))
    else:
        candidates.sort(key=lambda c:(c[0],c[1]),reverse=True)
    chosen=[];total=0
    for clear,duration,cue,a,b in candidates:
        if any(min(b,y)-max(a,x)>0 for _,_,_,x,y in chosen):continue
        # Unclear transcript references use voice-only conditioning. Prefer a
        # clean multi-utterance reference when adequate clear speech exists.
        if chosen and not clear and total>=rate:continue
        extra=b-a+(round(.12*rate) if chosen else 0)
        if total+extra>round(limit*rate):continue
        chosen.append((clear,duration,cue,a,b));total+=extra
        if total>=5*rate:break
    if not chosen:raise ValueError('No bounded solo actor reference available')
    chosen.sort(key=lambda c:c[3])
    power=sum((np.sum(speech[a:b]**2,axis=0) for _,_,_,a,b in chosen))
    channel=int(np.argmax(power))
    parts=[]
    for _,_,_,a,b in chosen:
        if parts:parts.append(np.zeros(round(.12*rate),dtype=np.float32))
        parts.append(speech[a:b,channel])
    reference=np.concatenate(parts)
    validate_reference_audio(reference,rate,limit=limit)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    sf.write(path,reference,rate,subtype='FLOAT')
    result={'text':' '.join(c['text'] for clear,_,c,_,_ in chosen) if all(x[0] for x in chosen) else '',
            'intervals':[[origin+a/rate,origin+b/rate] for _,_,_,a,b in chosen],
            'seconds':len(reference)/rate,'voiced_seconds':sum(x[1] for x in chosen),'channel':channel}
    if by_loudness:
        rows=[]
        for _,_,c,a,b in chosen:
            value=measured.get(id(c))
            rows.append({'cue':c.get('id'),'source_db':value,'delta_db':None if value is None else round(abs(value-target_db),6),
                         'interval':[origin+a/rate,origin+b/rate]})
        result['loudness']={'version':'reference-loudness-order-v1','target_db':target_db,'chosen':rows,
                            'unmeasured_candidates':sum(1 for v in measured.values() if v is None)}
    return result
