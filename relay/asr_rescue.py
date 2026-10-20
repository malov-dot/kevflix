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


def merge_rescue(original,rescued,clip_offset,*,whisper_no_speech=False):
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
                'text':' '.join(w['text'].strip() for w in group),'words':group,'recognition_pass':'caption-gap-no-vad'})
            covered.extend((w['start'],w['end']) for w in group)
    return sorted([*original,*additions],key=lambda s:(s['start'],s['end'])),len(additions)
