"""Extract explicit subtitle speaker evidence without assigning acoustic identity.

Input is the parsed plaintext caption shape returned by media.subtitle_cues.
An explicit label is an unverified textual assertion. A dash marks a possible
turn, never a speaker identity. Dialogue mentions are not searched for names.
"""
from __future__ import annotations

import math
import re


_PARTICLES = {'de', 'del', 'da', 'di', 'do', 'dos', 'du', 'la', 'le', 'van', 'von', 'of', 'the'}
_ROLES = {'man', 'woman', 'boy', 'girl', 'child', 'children', 'narrator', 'announcer',
          'speaker', 'voice', 'crowd', 'all', 'both', 'chorus', 'radio announcer',
          'male voice', 'female voice', 'unknown voice', 'offscreen voice'}
_NON_SPEAKER_WORDS = {
    'music', 'musical', 'instrumental', 'singing', 'sings', 'whispers', 'whispering',
    'yells', 'yelling', 'breathing', 'cough', 'coughs', 'coughing', 'wind', 'door',
    'doors', 'slams', 'slamming', 'footsteps', 'phone', 'telephone', 'rings',
    'ringing', 'engine', 'gunshot', 'gunshots', 'explosion', 'explosions',
    'applause', 'laughs', 'laughing', 'laughter', 'screams', 'screaming', 'sighs',
    'sighing', 'gasps', 'gasping', 'sobbing', 'crying', 'inaudible', 'indistinct',
    'speaking', 'foreign', 'language', 'unintelligible', 'silence', 'static',
    'english', 'spanish', 'french', 'german', 'italian', 'russian', 'japanese',
    'chinese', 'arabic', 'portuguese', 'korean', 'latin',
}
_NON_LABEL_PREFIXES = {'http', 'https', 'ftp', 'mailto', 'www', 'note', 'notes',
                       'warning', 'time', 'date', 'scene', 'chapter', 'caption',
                       'subtitles', 'title', 'example', 'translation'}
_NARRATIVE_STARTS = {'i', 'you', 'he', 'she', 'they', 'we', 'it', 'this', 'that',
                     'remember', 'meet', 'ask', 'tell', 'called', 'named', 'about',
                     'dear', 'hello', 'hey', 'please', 'meanwhile', 'earlier', 'later',
                     'yesterday', 'tomorrow'}
_DASH = re.compile(r'[-\u2013\u2014](?:[ \t]+|(?=[^\W\d_]))', re.UNICODE)
_WORD = re.compile(r"[^\W\d_]+(?:[’'\-][^\W\d_]+)*\.?", re.UNICODE)


def _plausible_explicit_label(value):
    """Conservative syntax filter, not a name recognizer or actor database."""
    label = value.strip()
    if not label or len(label) > 80 or any(ord(char) < 32 for char in label):
        return False
    lowered = ' '.join(label.casefold().split())
    if lowered in _NON_LABEL_PREFIXES or lowered.startswith(('in ', 'speaks ', 'speaking ')):
        return False
    words = label.split()
    if words[0].casefold() in _NARRATIVE_STARTS:
        return False
    if len(words) > 5 or any(word.casefold().strip('.') in _NON_SPEAKER_WORDS for word in words):
        return False
    if lowered in _ROLES:
        return True
    # Names may carry a numbered role (MAN 2), but a clock/time or URL cannot
    # pass this syntax. Number-only or ordinary prose prefixes are rejected.
    if words[-1].isdigit() and ' '.join(words[:-1]).casefold() in _ROLES:
        return True
    for index, word in enumerate(words):
        if not _WORD.fullmatch(word):
            return False
        if word.casefold().strip('.') in _PARTICLES and index > 0:
            continue
        if not word[0].isupper():
            return False
    return True


def caption_speaker_evidence(caption, *, source=None, caption_index=None, origin=0.0):
    """Return explicit label/turn observations with exact text provenance.

    Times are caption start/end plus origin; character spans are zero-based,
    half-open offsets in the original field. Labels retain case/diacritics and
    raw_label retains surrounding whitespace. Actor fields and body labels are
    both emitted even when they conflict. No precedence or voice binding occurs.
    """
    if not isinstance(caption, dict):
        raise TypeError('caption must be a parsed caption dictionary')
    start, end = float(caption['start']) + origin, float(caption['end']) + origin
    if not all(math.isfinite(value) for value in (start, end)) or start < 0 or end <= start:
        raise ValueError('Caption evidence requires a finite positive time interval')
    text = caption.get('text', '')
    if not isinstance(text, str):
        raise TypeError('caption text must be plaintext')
    source = source if source is not None else caption.get('source')
    evidence = []

    def emit(kind, label, raw_label, marker, field, begin, stop, line, syntax):
        evidence.append(dict(kind=kind, label=label, raw_label=raw_label, marker=marker,
                             start=start, end=end, candidate_only=True,
                             acoustic_identity=None, requires_identity_anchor=label is not None,
                             provenance=dict(source=source, caption_index=caption_index,
                                             caption_id=caption.get('id'), field=field,
                                             character_start=begin, character_end=stop,
                                             line=line, syntax=syntax)))

    actor = caption.get('actor')
    if isinstance(actor, str) and actor.strip() and len(actor) <= 256 and not any(ord(c) < 32 for c in actor):
        # This is a structured subtitle field, so preserve it even if it does
        # not look like an English name. Its truth still needs an audio anchor.
        emit('actor_field', actor.strip(), actor, actor, 'actor', 0, len(actor), None, 'parsed_actor_field')

    offset = 0
    for line_index, full_line in enumerate(text.splitlines(keepends=True)):
        line = full_line.rstrip('\r\n')
        position = len(line) - len(line.lstrip(' \t'))
        turn = _DASH.match(line, position)
        if turn:
            # No anonymous actor ID is invented for an unlabeled dash turn.
            emit('turn_marker', None, None, line[position:turn.end()], 'text',
                 offset+position, offset+turn.end(), line_index, 'dash_prefix')
            position = turn.end()
        remaining = line[position:]
        if remaining.startswith('['):
            close = remaining.find(']')
            if close > 1:
                raw = remaining[1:close]
                tail = remaining[close+1:].strip()
                # A bare stage direction or bare name is not attributed speech.
                if tail and _plausible_explicit_label(raw):
                    emit('explicit_caption_label', raw.strip(), raw, remaining[:close+1], 'text',
                         offset+position, offset+position+close+1, line_index, 'bracket_prefix')
        else:
            colon = remaining.find(':')
            if 0 < colon <= 80:
                raw, tail = remaining[:colon], remaining[colon+1:]
                # Require a visible whitespace separator, excluding URI/time
                # syntax and embedded punctuation in ordinary dialogue.
                if tail and tail[0].isspace() and tail.strip() and _plausible_explicit_label(raw):
                    emit('explicit_caption_label', raw.strip(), raw, remaining[:colon+1], 'text',
                         offset+position, offset+position+colon+1, line_index, 'colon_prefix')
        offset += len(full_line)
    return evidence


_STAGE_ONLY = re.compile(r'\s*(?:\[[^\]]*\]|\([^)]*\))\s*')


def implied_speaker_turns(caption, evidence=None):
    """Count speaker segments implied by caption layout; >1 marks a mixed cue.

    A dash or explicit body label on a later dialogue line starts a new
    segment even when the first line is unmarked ("Hello.\\n- Bye."). Lines
    that are only bracketed/parenthesized stage directions are ignored. This
    is a conservative abstention signal, never speaker identification.
    """
    text = caption.get('text', '')
    if not isinstance(text, str):
        raise TypeError('caption text must be plaintext')
    if evidence is None:
        evidence = caption_speaker_evidence(caption)
    lines = [index for index, line in enumerate(text.splitlines())
             if line.strip() and not _STAGE_ONLY.fullmatch(line)]
    if not lines:
        return 0
    boundaries = {item['provenance']['line'] for item in evidence
                  if item['provenance'].get('field') == 'text'
                  and item['kind'] in ('turn_marker', 'explicit_caption_label')} & set(lines)
    return len(boundaries) + (lines[0] not in boundaries)


def extract_caption_speakers(captions, *, source=None, origin=0.0):
    """Batch extraction; list order is the caption provenance index."""
    return [item for index, caption in enumerate(captions)
            for item in caption_speaker_evidence(caption, source=source,
                                                 caption_index=index, origin=origin)]
