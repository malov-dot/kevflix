"""Recover audio-backed caption gaps without replacing recognized dialogue."""
from __future__ import annotations
from .dialogue import caption_text,is_no_speech


def union_seconds(intervals):
    total=0.;right=None
    for a,b in sorted((a,b) for a,b in intervals if b>a):
        if right is None or a>right:total+=b-a;right=b
        elif b>right:total+=b-right;right=b
    return total


def rescue_windows(segments,turns,captions,origin,end,*,minimum_coverage=.65,max_seconds=20,max_windows=3,whisper_no_speech=False):
    recognized=[]
    for segment in segments:
        if is_no_speech(segment,whisper_no_speech):continue
        for word in segment.get('words') or [segment]:
            if word.get('probability',1)>=.25:recognized.append((origin+word['start'],origin+word['end']))
    windows=[]
    for caption in captions:
        if not caption_text(caption['text']):continue
        speech=[(max(origin,caption['start'],t['start']),min(end,caption['end'],t['end'])) for t in turns]
        voiced=union_seconds(speech)
        if voiced<.25:continue
        covered=union_seconds((max(a,c),min(b,d)) for a,b in speech for c,d in recognized)
        if covered/voiced>=minimum_coverage:continue
        a=max(origin,caption['start']-.35);b=min(end,caption['end']+.35)
        if b>a:windows.append((a,b))
    merged=[]
    for a,b in sorted(windows):
        if merged and a-merged[-1][1]<=1.2 and b-merged[-1][0]<=max_seconds:
            merged[-1]=(merged[-1][0],max(merged[-1][1],b))
        else:merged.append((a,min(b,a+max_seconds)))
    return merged[:max_windows]


SPEECH_GAP_MIN_SECONDS=1.0       # an untranscribed stretch of the detector's speaking windows this long is worth a second look
SPEECH_GAP_WORD_PAD=.2           # a recognized word covers this much around itself
SPEECH_GAP_MIN_SPEECH=.30        # classifier 'Speech' score a gap needs before it is read (97 % of a full film's dubbed lines reach it)


def speech_like(gaps,labels,minimum=SPEECH_GAP_MIN_SPEECH):
    """The gaps whose classifier labels ({label: probability} per gap, or None) carry 'Speech' at or above `minimum`.
    A gap without labels is dropped: reading audio that no witness calls speech invents text over music."""
    kept=[]
    for gap,events in zip(gaps,labels or ()):
        score=events.get('Speech') if isinstance(events,dict) else None
        if isinstance(score,(int,float)) and not isinstance(score,bool) and score>=minimum:kept.append(gap)
    return kept


def speech_gap_windows(segments,turns,origin,end,*,render_start=None,render_end=None,minimum_seconds=SPEECH_GAP_MIN_SECONDS,
                       pad=.35,max_seconds=20,max_windows=3,whisper_no_speech=False):
    """F16: where to listen again when there are no captions to point at missed speech.

    Stretches of the detector's speaking windows (`turns`, absolute seconds, any speaker) of at least `minimum_seconds` that no
    recognized word (probability >= .25, +-SPEECH_GAP_WORD_PAD) covers. Only stretches that START inside
    [render_start, render_end) count: the window that owns that speech looks, its neighbours do not. The `max_windows` longest
    are returned as [(start, end)] absolute, padded by `pad`, clipped to [origin, end], each at most `max_seconds`, in time
    order with overlapping ones joined. Segment times are relative to `origin` (as in rescue_windows)."""
    low=origin if render_start is None else float(render_start)
    high=end if render_end is None else float(render_end)
    recognized=[]
    for segment in segments:
        if is_no_speech(segment,whisper_no_speech):continue
        for word in segment.get('words') or [segment]:
            if word.get('probability',1)>=.25:
                recognized.append((origin+word['start']-SPEECH_GAP_WORD_PAD,origin+word['end']+SPEECH_GAP_WORD_PAD))
    recognized.sort()
    speaking=[];right=None
    for a,b in sorted((max(origin,float(t['start'])),min(end,float(t['end']))) for t in turns):
        if b<=a:continue
        if right is None or a>right:speaking.append([a,b]);right=b
        elif b>right:speaking[-1][1]=b;right=b
    gaps=[]
    for a,b in speaking:
        cursor=a
        for c,d in recognized:
            if d<=cursor or c>=b:continue
            if c>cursor:gaps.append((cursor,min(c,b)))
            cursor=max(cursor,d)
            if cursor>=b:break
        if cursor<b:gaps.append((cursor,b))
    gaps=[(a,b) for a,b in gaps if b-a>=minimum_seconds and low<=a<high]
    chosen=sorted(sorted(gaps,key=lambda g:(g[0]-g[1],g[0]))[:max_windows])
    merged=[]
    for a,b in chosen:
        a=max(origin,a-pad);b=min(end,b+pad,a+max_seconds)
        if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],min(max(merged[-1][1],b),merged[-1][0]+max_seconds))
        elif b>a:merged.append((a,b))
    return merged


def merge_rescue(original,rescued,clip_offset,*,whisper_no_speech=False,label='caption-gap-no-vad'):
    """Retain original words; add only well-supported uncovered rescue words."""
    covered=[]
    for segment in original:
        if is_no_speech(segment,whisper_no_speech):continue
        for word in segment.get('words') or [segment]:covered.append((word['start'],word['end']))
    additions=[]
    for segment in rescued:
        if is_no_speech(segment,whisper_no_speech) or segment.get('log_probability',0)<-1.2:continue
        groups=[]
        for word in segment.get('words') or []:
            current={**word,'start':word['start']+clip_offset,'end':word['end']+clip_offset}
            a,b=current['start'],current['end']
            occupied=union_seconds((max(a,c),min(b,d)) for c,d in covered)
            if b<=a or current.get('probability',0)<.3 or occupied/(b-a)>.35:
                if groups and groups[-1]:groups.append([])
                continue
            if not groups or (groups[-1] and a-groups[-1][-1]['end']>.65):groups.append([])
            groups[-1].append(current)
        for group in groups:
            if not group:continue
            additions.append({**segment,'start':group[0]['start'],'end':group[-1]['end'],
                'text':' '.join(w['text'].strip() for w in group),'words':group,'recognition_pass':label})
            covered.extend((w['start'],w['end']) for w in group)
    return sorted([*original,*additions],key=lambda s:(s['start'],s['end'])),len(additions)
