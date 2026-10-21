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


PURITY_VERSION='reference-purity-v1'
PURITY_MIN_COSINE=.30
PURITY_SCOPES=('excerpt','reference')


def _unit_vector(vector):
    """``vector`` as an L2-normalised list of floats, or None when it is missing, non-finite or all zero."""
    if vector is None:return None
    try:values=[float(x) for x in vector]
    except (TypeError,ValueError,OverflowError):return None
    if not values or not all(math.isfinite(x) for x in values):return None
    norm=math.sqrt(sum(x*x for x in values))
    return [x/norm for x in values] if norm>=1e-12 else None


def _majority_label(raw_labels,seconds):
    """(majority raw label, 'ok'), or (None, why there is nothing to decide).

    Majority = the label of most excerpts; a count tie goes to the label with more audio; a tie in both is no majority."""
    if len(raw_labels)<2:return None,'single_excerpt'
    if any(not isinstance(label,str) or not label.strip() for label in raw_labels):return None,'labels_unavailable'
    totals={}
    for label,duration in zip(raw_labels,seconds):
        row=totals.setdefault(label,[0,0.]);row[0]+=1;row[1]+=float(duration)
    if len(totals)==1:return None,'single_label'
    ranked=sorted(totals.items(),key=lambda item:(-item[1][0],-item[1][1],item[0]))
    if ranked[0][1][0]==ranked[1][1][0] and abs(ranked[0][1][1]-ranked[1][1][1])<=1e-9:return None,'no_majority_label'
    return ranked[0][0],'ok'


def reference_purity_decision(raw_labels,seconds,vectors,*,min_cosine=PURITY_MIN_COSINE,scope='excerpt'):
    """Which excerpts of one reference are probably somebody else's voice. Pure; no audio, no model.

    ``raw_labels[i]`` / ``seconds[i]`` / ``vectors[i]``: the raw diarization label, duration and speaker embedding (or None)
    of excerpt i. scope='excerpt' (the default): excerpt i is dropped only when BOTH its own raw label differs from the
    majority raw label AND its embedding cosine to the centroid of the other excerpts' (unit-length) embeddings is under
    ``min_cosine``. scope='reference': the raw labels of the reference must not be unanimous (and must have a majority), and
    then every excerpt under ``min_cosine`` is dropped whatever its own label (a diarization label can itself be wrong, so
    the label-differing excerpt may be the pure one); if that is every excerpt, only the non-majority-label ones go.
    Anything unavailable (a label, the embeddings, a majority) drops nothing, and at least one excerpt always stays. Returns
    {'reason','majority_label','cosines' (every excerpt vs the centroid of the others, None when not computable),'drop'}."""
    if scope not in PURITY_SCOPES:raise ValueError('Unknown reference purity scope %r' % (scope,))
    n=len(raw_labels)
    out={'reason':None,'majority_label':None,'cosines':[None]*n,'drop':[]}
    majority,reason=_majority_label(raw_labels,seconds)
    if majority is None:out['reason']=reason;return out
    out['majority_label']=majority
    units=[_unit_vector(v) for v in vectors]
    if len({len(u) for u in units if u is not None})>1:out['reason']='embedding_dimension_mismatch';return out
    for i in range(n):
        others=[u for j,u in enumerate(units) if j!=i and u is not None]
        if units[i] is None or not others:continue
        centroid=[sum(column)/len(others) for column in zip(*others)]
        norm=math.sqrt(sum(x*x for x in centroid))
        if norm<1e-12:continue
        out['cosines'][i]=sum(a*b for a,b in zip(units[i],centroid))/norm
    if all(c is None for c in out['cosines']):out['reason']='embeddings_unavailable';return out
    low=[i for i in range(n) if out['cosines'][i] is not None and out['cosines'][i]<min_cosine]
    if scope=='reference':drop=[i for i in low if raw_labels[i]!=majority] if len(low)>=n else low
    else:drop=[i for i in low if raw_labels[i]!=majority]
    out['drop']=drop if len(drop)<n else []
    out['reason']='dropped' if out['drop'] else 'clean'
    return out


def _mono_16k(samples,rate):
    """float32 mono 16 kHz from a mono excerpt (the resampling relay.audio_evidence.to_mono_16k and the pipeline use)."""
    import numpy as np
    x=np.ascontiguousarray(samples,dtype=np.float32)
    if int(rate)!=16000 and len(x):
        from scipy.signal import resample_poly
        g=math.gcd(int(rate),16000)
        x=resample_poly(x,16000//g,int(rate)//g).astype(np.float32)
    return x


def _embedding_errors():
    """What an embedder may raise for 'no usable voice evidence' (-> fail open). Cancellation, resource waits and backend
    or configuration faults (every other MovieFault) are NOT swallowed."""
    try:
        from .movie_faults import MovieContentFallback
    except ImportError:
        return (ValueError,TypeError,ArithmeticError)
    return (MovieContentFallback,ValueError,TypeError,ArithmeticError)


def _purify_chosen(chosen,speech,origin,rate,raw_labels,embed,scope):
    """(the chosen excerpts that stay, the ``purity`` receipt). Embeds only when the raw labels differ (clean references
    cost nothing); the excerpts are embedded exactly as they are cut into the reference (same channel, 16 kHz mono)."""
    import numpy as np
    ordered=sorted(chosen,key=lambda c:c[3])
    labels=[raw_labels.get(id(c[2])) for c in ordered]
    seconds=[(c[4]-c[3])/rate for c in ordered]
    receipt={'version':PURITY_VERSION,'scope':scope,'threshold':PURITY_MIN_COSINE,'checked':False,'reason':None,
             'majority_label':None,'excerpts':[],'dropped':[]}
    def note(drop=(),cosines=()):
        for i,(_,_,cue,a,b) in enumerate(ordered):
            row={'cue':cue.get('id'),'interval':[origin+a/rate,origin+b/rate],'raw_label':labels[i],
                 'seconds':round(seconds[i],3),'cosine_to_others':None if not cosines or cosines[i] is None else round(cosines[i],4),
                 'dropped':i in drop}
            receipt['excerpts'].append(row)
            if i in drop:receipt['dropped'].append({k:row[k] for k in ('cue','interval','raw_label','seconds','cosine_to_others')})
    majority,reason=_majority_label(labels,seconds)
    if majority is None:
        receipt['reason']=reason;note();return chosen,receipt
    channel=int(np.argmax(sum(np.sum(speech[a:b]**2,axis=0) for _,_,_,a,b in ordered)))
    errors=_embedding_errors()
    vectors=[]
    for _,_,_,a,b in ordered:
        try:vectors.append(embed(_mono_16k(speech[a:b,channel],rate)))
        except errors:vectors.append(None)
    decision=reference_purity_decision(labels,seconds,vectors,scope=scope)
    receipt.update(checked=decision['reason'] in ('clean','dropped'),reason=decision['reason'],majority_label=decision['majority_label'])
    note(decision['drop'],decision['cosines'])
    if not decision['drop']:return chosen,receipt
    return [c for i,c in enumerate(ordered) if i not in decision['drop']],receipt


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
    sorts after every measured one. The result gains a ``loudness`` receipt; no gain is ever applied to the audio.

    This function never checks that the chosen excerpts are one person; see make_pure_reference for that."""
    return _build_reference(speech,cues,origin,rate,path,limit=limit,speech_regions=speech_regions,
                            loudness_target_db=loudness_target_db,loudness_db=loudness_db,embed=None,purity_scope='excerpt')


def make_pure_reference(speech,cues,origin,rate,path,*,limit=8,speech_regions=None,loudness_target_db=None,loudness_db=None,embed=None,
                        purity_scope='excerpt'):
    """make_reference plus the reference-purity check (``embed`` given), for bank references joined from several excerpts.

    ``embed(samples)``: 16 kHz mono float32 -> speaker embedding (list of floats), or None when there is no usable voice
    evidence; in the pipeline this is ``lambda x: self._embedding(owner, x)``, the Community-1 embedding every other speaker
    decision already uses (no new model). With ``embed=None`` this IS make_reference (same result, no ``purity`` key).
    With it, the selected excerpts are compared after selection (reference_purity_decision): an excerpt is dropped only when
    BOTH its raw diarization label differs from the majority raw label of the excerpts AND its embedding cosine to the
    centroid of the other excerpts is under PURITY_MIN_COSINE (``purity_scope='reference'``: see reference_purity_decision).
    It fails open (nothing dropped) without labels, embeddings or a majority, and never leaves fewer than one excerpt. The reference WAV, ``intervals``, ``text`` and ``voiced_seconds``
    describe the KEPT excerpts only, so the source-bound admission still sees an exact stem excerpt; what was dropped is in
    the additive ``purity`` receipt of the result."""
    if purity_scope not in PURITY_SCOPES:raise ValueError('Unknown reference purity scope %r' % (purity_scope,))
    return _build_reference(speech,cues,origin,rate,path,limit=limit,speech_regions=speech_regions,
                            loudness_target_db=loudness_target_db,loudness_db=loudness_db,embed=embed,purity_scope=purity_scope)


def _build_reference(speech,cues,origin,rate,path,*,limit,speech_regions,loudness_target_db,loudness_db,embed,purity_scope):
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
    raw_labels={}
    candidates=[]
    for cue in cues:
        if cue.get('overlapped') or cue.get('turn_overlap') is True:continue
        begin=cue.get('speech_start',cue['start'])
        end=cue.get('speech_end',cue['end'])
        if not .25<=end-begin<=limit:continue
        a=max(0,round((begin-origin-.06)*rate));b=min(len(speech),round((end-origin+.06)*rate))
        if b<=a:continue
        solo=excerpt_solo_label(speech_regions,(origin+a/rate,origin+b/rate),(begin,end))
        if solo is None:continue
        excerpt=speech[a:b]
        if not np.all(np.isfinite(excerpt)):continue
        # A quiet or invalid early cue must not displace a later usable source excerpt. The final joined mono
        # reference is validated again, including the renderer's full-duration RMS and minimum length.
        if float(np.max(np.sqrt(np.mean(np.square(excerpt,dtype=np.float64),axis=0))))<=MIN_REFERENCE_RMS:continue
        candidates.append((not cue.get('unclear',False),end-begin,cue,a,b))
        raw_labels[id(cue)]=solo
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
    purity=None
    if embed is not None:chosen,purity=_purify_chosen(chosen,speech,origin,rate,raw_labels,embed,purity_scope)
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
    if purity is not None:result['purity']=purity
    if by_loudness:
        rows=[]
        for _,_,c,a,b in chosen:
            value=measured.get(id(c))
            rows.append({'cue':c.get('id'),'source_db':value,'delta_db':None if value is None else round(abs(value-target_db),6),
                         'interval':[origin+a/rate,origin+b/rate]})
        result['loudness']={'version':'reference-loudness-order-v1','target_db':target_db,'chosen':rows,
                            'unmeasured_candidates':sum(1 for v in measured.values() if v is None)}
    return result
