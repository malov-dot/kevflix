"""Audio-led dialogue planning with caption hints and explicit unresolved cases."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import math
import re


def overlap(a, b):
    return max(0.0, min(a['end'], b['end']) - max(a['start'], b['start']))


def _positive_interval(start, end):
    """Observed timestamps only: never repair an impossible interval by inventing duration."""
    if isinstance(start, bool) or isinstance(end, bool):
        return False
    try:
        return math.isfinite(start) and math.isfinite(end) and start >= 0 and end > start
    except (TypeError, ValueError, OverflowError):
        return False


# Whisper's own skip rule (openai-whisper transcribe.py, faster-whisper
# transcribe.py) drops a segment only when no_speech_prob > .6 AND
# avg_logprob < -1.0. The legacy relay rule looks at no_speech alone.
WHISPER_NO_SPEECH_THRESHOLD=.6
WHISPER_LOGPROB_THRESHOLD=-1.0


def is_no_speech(segment, whisper_rule=False):
    """Legacy rule (default): no_speech_probability > .6 alone.

    whisper_rule=True (beta first-pass lane only): also require
    log_probability < -1.0; a segment without log_probability keeps the
    legacy outcome (dropped), so missing evidence never keeps a segment.
    """
    if not segment.get('no_speech_probability',0)>WHISPER_NO_SPEECH_THRESHOLD:return False
    if not whisper_rule:return True
    value=segment.get('log_probability')
    return value is None or value<WHISPER_LOGPROB_THRESHOLD


def caption_edges(captions):
    """Start/end times of captions that carry text (movie timeline), sorted."""
    return sorted({float(t) for c in captions if caption_text(c['text']) for t in (c['start'],c['end'])})


def caption_text(text):
    # Keep sung words; remove only explicitly bracketed nonverbal descriptions.
    value = re.sub(r'\[[^\]]*\]|\([^)]*\)', '', str(text))
    return re.sub(r'\s+', ' ', value.replace('♪', '').replace('♫', '')).strip()


def plan_dialogue(segments, turns, captions=(), *, origin=0.0, minimum_probability=.65, speech_regions=None,
                  whisper_no_speech=False, caption_edge_pause=None, overlap_fraction=.25, word_floor=.5, low_word_share=0.0):
    """Never use an entire caption list in place of uncovered recognized speech.

    Times in segments are relative; turns and captions use the movie timeline.
    A caption is a translation fallback, not proof that speech occurred.

    Beta-only options (defaults reproduce the legacy plan exactly):
    whisper_no_speech  drop a segment only by Whisper's two-part rule (is_no_speech).
    caption_edge_pause also start a new group at an inter-word pause of at least
                       this many seconds that contains a caption start or end.
    """
    groups, discarded, issues = [], [], []
    edges = caption_edges(captions) if caption_edge_pause is not None else ()
    def edge_pause(left, right):
        return right-left >= caption_edge_pause and any(left <= t <= right for t in edges)
    for segment in segments:
        if is_no_speech(segment, whisper_no_speech):
            discarded.append({'reason':'asr_no_speech', 'text':segment['text']})
            continue
        tokens = segment.get('words') or [dict(segment, probability=0)]
        new_segment=True
        for token in tokens:
            if not _positive_interval(token.get('start'), token.get('end')):
                discarded.append({'reason':'asr_invalid_interval', 'text':token.get('text',''),
                                  'start':token.get('start'), 'end':token.get('end'),
                                  'time_basis':'analysis-relative-seconds'})
                continue
            word = {**token, 'start':token['start']+origin, 'end':token['end']+origin}
            if not _positive_interval(word['start'], word['end']):
                discarded.append({'reason':'asr_invalid_interval', 'text':word.get('text',''),
                                  'start':word['start'], 'end':word['end'],
                                  'time_basis':'absolute-source-seconds'})
                continue
            if not word.get('text','').strip():
                continue
            coverage = defaultdict(float)
            for turn in turns:
                coverage[turn['speaker']] += overlap(word,turn)
            ranked = sorted(coverage.items(),key=lambda x:x[1],reverse=True)
            speaker = ranked[0][0] if ranked and ranked[0][1] > 0 else None
            uncertain_overlap = len(ranked)>1 and ranked[1][1] > overlap_fraction*(word['end']-word['start'])
            key = (speaker,uncertain_overlap)
            if new_segment or not groups or groups[-1]['key'] != key or word['start']-groups[-1]['end'] > .65 or groups[-1]['words'][-1]['text'].strip().endswith(('.', '!', '?')) or (
                    edges and edge_pause(groups[-1]['end'], word['start'])):
                groups.append({'key':key,'start':word['start'],'end':word['end'], 'words':[]})
            groups[-1]['words'].append(word)
            groups[-1]['start']=min(groups[-1]['start'],word['start'])
            groups[-1]['end']=max(groups[-1]['end'],word['end'])
            new_segment=False
    # Caption timestamps may begin halfway through an ASR word. Anchor whole
    # spoken phrases, never split a sentence at an approximate caption boundary.
    merged=[]
    # Rescue segments are sorted by their PARENT start, but their words can
    # precede a late word group from an overlapping original segment. Merge in
    # actual group-time order, never parent traversal order: otherwise a later
    # group can retain its start while an earlier rescue overwrites its end.
    for group in sorted(groups,key=lambda g:(g['start'],g['end'])):
        matching=[(overlap(group,c),i) for i,c in enumerate(captions) if caption_text(c['text'])]
        match=max(matching,default=(0,None))
        caption=match[1] if match[0]>0 else None
        speaker,overlapped=group['key']
        group['key']=(speaker,caption,overlapped)
        if merged and group['key']==merged[-1]['key'] and caption is not None and group['start']-merged[-1]['end']<=.65:
            merged[-1]['words'].extend(group['words'])
            merged[-1]['end']=max(merged[-1]['end'],group['end'])
        else:merged.append(group)
    groups=merged
    result=[]
    for group in groups:
        if not _positive_interval(group['start'],group['end']):
            discarded.append({'reason':'asr_invalid_group_interval',
                              'text':' '.join(x.get('text','').strip() for x in group['words']).strip(),
                              'start':group['start'],'end':group['end'],
                              'time_basis':'absolute-source-seconds'})
            continue
        speaker,caption_index,overlapped=group.pop('key')
        # Keep the recognized/rescued words in their observed temporal order;
        # nested groups must not move an earlier word behind a later one.
        tokens=group['words']=sorted(group['words'],key=lambda word:(word['start'],word['end']))
        text=' '.join(x['text'].strip() for x in tokens).strip()
        probability=sum(x.get('probability',0) for x in tokens)/len(tokens)
        ident=hashlib.sha256(('%0.3f:%0.3f:%s'%(group['start'],group['end'],text)).encode()).hexdigest()[:24]
        cue={**group,'id':ident,'text':text,'speaker':speaker,'probability':probability,
             'basis':'audio','caption_index':caption_index,'overlapped':overlapped}
        support=[t for t in turns if t['speaker']==speaker and overlap(group,t)>0]
        if support:
            cue['speech_start']=min(t.get('speech_start',t['start']) for t in support)
            cue['speech_end']=max(t.get('speech_end',t['end']) for t in support)
        if caption_index is not None:
            cue['caption']=dict(captions[caption_index])
            cue['singing']=bool(re.search('[♪♫]',captions[caption_index]['text']))
        if speaker is None:issues.append({'kind':'speaker_unresolved','cue':ident})
        if overlapped:issues.append({'kind':'overlapping_dialogue','cue':ident})
        # low_word_share: the share of a cue's words allowed below word_floor (0 = none, the earlier rule). One mumbled
        # word in a long sentence does not make the sentence unreadable; most of its words doing so does.
        low=sum(1 for x in tokens if x.get('probability',0)<word_floor)
        if probability<minimum_probability or low>low_word_share*len(tokens):cue['unclear']=True
        result.append(cue)
    # Captions without recognized words can still cover unclear audible speech.
    # Preserve these for explicit resolution instead of silently losing a line.
    used={c['caption_index'] for c in result}
    for index,caption in enumerate(captions):
        if index in used:continue
        text=caption_text(caption['text'])
        matching=[t for t in (speech_regions if speech_regions is not None else turns) if overlap(t,caption)>.1]
        if text and matching:
            issues.append({'kind':'speech_without_transcript','caption':dict(caption),
                           'speakers':sorted({t['speaker'] for t in matching})})
        else:
            discarded.append({'reason':'caption_without_speech','text':caption['text']})
    # Several speaker turns in a caption must not each repeat its entire text.
    by_caption=defaultdict(list)
    for cue in result:
        if cue['caption_index'] is not None:by_caption[cue['caption_index']].append(cue)
    for index,cues in by_caption.items():
        parts=[caption_text(x) for x in re.split(r'(?:^|\s)-\s+',captions[index]['text']) if caption_text(x)]
        if len(parts)!=len(cues) and len(cues)>1:
            sentences=[x.strip() for x in re.split(r'(?<=[.!?])\s+',caption_text(captions[index]['text'])) if x.strip()]
            if len(sentences)==len(cues):parts=sentences
        if len(cues)==1:
            cues[0]['caption_fallback']=caption_text(captions[index]['text'])
        elif len(parts)==len(cues) and len(parts)>1:
            for cue,part in zip(cues,parts):cue['caption_fallback']=part
        for cue in cues:
            if cue.get('unclear') and not cue.get('caption_fallback'):
                issues.append({'kind':'unclear_dialogue_without_aligned_caption','cue':cue['id']})
    return result,issues,discarded
