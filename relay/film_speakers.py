"""Film-level speakers (F15, 2026-11-11): one id per person for the whole film, from Diart run ahead of playback.

Diart stays the detector. A second Diart stream (the "scout") is fed the film's dialogue channel ahead of playback
(relay/film_scout.py). Its labels are consolidated here into film speakers S01, S02, ... and every window's own live Diart
labels are mapped onto them by who speaks when. Nothing in this module loads a model.

Why the consolidation is this careful (measured on a full film, 38 live labels, stored speaker vectors): the nearest labels by
embedding are NOT reliably one person. Of the 11 merges cosine distance proposes at 0.4, 3 join a low and a high voice and 4
join two labels that talk over each other. So two labels are merged only when
  * the cosine distance between their Diart centres is at most MERGE_DISTANCE under COMPLETE linkage (every member of one
    group is that close to every member of the other: no chaining), and
  * they never speak at the same time (TOGETHER_SECONDS in total), and
  * their median pitch, where both are known, is within PITCH_SEMITONES.
An unsure pair stays two speakers. Because a film speaker can still be two people, a film id may only TIGHTEN a voice decision
(refuse a donor), never loosen one (see same_speaker). The line's own audio remains the only source of its voice.

Artifact (<cache>/<job>/film-speakers.json):
  {'version', 'covered_until': film seconds the scout has analysed, 'labels': {scout label: film id},
   'speakers': {film id: {'labels', 'seconds', 'pitch_hz', 'first'}}, 'turns': [[start, end, scout label], ...],
   'refused': [{'a', 'b', 'distance', 'reason'}], 'thresholds': {...}}
Standard library only (centres are plain lists of floats).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

VERSION = 'film-speakers-v1'
FILE_NAME = 'film-speakers.json'
MERGE_DISTANCE = .35          # complete-linkage cosine distance at or under which two scout labels may be one speaker
PITCH_SEMITONES = 5.0         # actor_pipeline.REFERENCE_DONOR_PITCH_SEMITONES: further apart is a different-sounding voice
TOGETHER_SECONDS = 1.0        # two labels that overlap this long in total are two people
MAP_MIN_SHARE = .6            # a live label maps to a film speaker that covers at least this share of its speech in the window
MAP_MIN_SECONDS = .5          # ... and at least this much of it
ID_PREFIX = 'S'


def enabled(beta):
    """actor_beta.film_speakers (default on with the beta lane; only an explicit false turns it off)."""
    return isinstance(beta, dict) and beta.get('enabled') is True and beta.get('film_speakers') is not False


def path_for(job_folder):
    return Path(job_folder) / FILE_NAME


def cosine_distance(a, b):
    """1 - cosine similarity of two equal-length vectors; 2.0 (as far as possible) when either is empty or zero."""
    if not a or not b or len(a) != len(b):
        return 2.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na, nb = math.sqrt(sum(float(x) ** 2 for x in a)), math.sqrt(sum(float(y) ** 2 for y in b))
    if na < 1e-12 or nb < 1e-12 or not math.isfinite(dot):
        return 2.0
    return 1.0 - dot / (na * nb)


def semitones(a, b):
    return abs(12.0 * math.log2(float(a) / float(b)))


def pair_key(a, b):
    return '|'.join(sorted((str(a), str(b))))


def together_seconds(turns):
    """{pair_key: seconds} two different labels are active at the same time. turns: [(start, end, label)] or
    [{'start','end','speaker'}]."""
    rows = sorted((float(t[0]), float(t[1]), str(t[2])) if not isinstance(t, dict) else
                  (float(t['start']), float(t['end']), str(t['speaker'])) for t in turns)
    out = {}
    for i, (a, b, x) in enumerate(rows):
        for c, d, y in rows[i + 1:]:
            if c >= b:
                break
            if x != y:
                shared = min(b, d) - c
                if shared > 0:
                    key = pair_key(x, y)
                    out[key] = out.get(key, 0.0) + shared
    return out


def cannot_link(a, b, prints, together, *, pitch_semitones=PITCH_SEMITONES, together_limit=TOGETHER_SECONDS):
    """The reason two labels are certainly two people, or None."""
    if together.get(pair_key(a, b), 0.0) >= together_limit:
        return 'speak_at_the_same_time'
    pa, pb = (prints.get(a) or {}).get('pitch_hz'), (prints.get(b) or {}).get('pitch_hz')
    if pa and pb and semitones(pa, pb) > pitch_semitones:
        return 'pitch_apart'
    return None


def consolidate(prints, together=None, *, merge_distance=MERGE_DISTANCE, pitch_semitones=PITCH_SEMITONES,
                together_limit=TOGETHER_SECONDS):
    """Groups of scout labels that are one film speaker.

    prints: {label: {'centre': [float, ...], 'seconds': float, 'pitch_hz': float | None, 'first': float}}.
    together: {pair_key(a, b): seconds both spoke at once}. Complete linkage: the closest pair of groups is merged while
    EVERY cross pair of labels is within ``merge_distance`` and no cross pair is a cannot-link. Returns
    (groups, refused): groups = [[label, ...]] (each sorted), refused = [{'a', 'b', 'distance', 'reason'}] for label pairs
    that were close enough but may not be merged."""
    together = together or {}
    labels = sorted(prints)
    distance = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            distance[pair_key(a, b)] = cosine_distance(prints[a].get('centre'), prints[b].get('centre'))
    refused = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            if distance[pair_key(a, b)] <= merge_distance:
                reason = cannot_link(a, b, prints, together, pitch_semitones=pitch_semitones, together_limit=together_limit)
                if reason:
                    refused.append({'a': a, 'b': b, 'distance': round(distance[pair_key(a, b)], 4), 'reason': reason})
    blocked = {pair_key(r['a'], r['b']) for r in refused}
    groups = [[label] for label in labels]
    while True:
        best = None
        for x in range(len(groups)):
            for y in range(x + 1, len(groups)):
                pairs = [pair_key(a, b) for a in groups[x] for b in groups[y]]
                if any(key in blocked for key in pairs):
                    continue
                worst = max(distance[key] for key in pairs)
                if worst <= merge_distance and (best is None or worst < best[0]):
                    best = (worst, x, y)
        if best is None:
            break
        _, x, y = best
        groups[x] = sorted(groups[x] + groups[y])
        del groups[y]
    return groups, refused


def name_groups(groups, prints, previous=None):
    """{label: film id}. A group keeps the lowest id any of its labels had in ``previous`` (ids never change for a label
    unless two speakers are found to be one, and then the lower id wins); other groups are numbered in order of first
    speech."""
    previous = previous or {}
    def number(value):
        try:
            return int(str(value)[len(ID_PREFIX):])
        except ValueError:
            return None
    taken, named, fresh = set(), {}, []
    for group in sorted(groups, key=lambda g: min((prints.get(l) or {}).get('first', 0.0) for l in g)):
        old = sorted(n for n in (number(previous[l]) for l in group if l in previous) if n is not None and n not in taken)
        if old:
            taken.add(old[0])
            for label in group:
                named[label] = old[0]
        else:
            fresh.append(group)
    counter = 1
    for group in fresh:
        while counter in taken:
            counter += 1
        taken.add(counter)
        for label in group:
            named[label] = counter
    return {label: '%s%02d' % (ID_PREFIX, n) for label, n in named.items()}


def build(prints, turns, covered_until, previous=None, **thresholds):
    """The artifact for what the scout knows so far. turns: [(start, end, scout label)]."""
    rows = sorted((round(float(a), 3), round(float(b), 3), str(label)) for a, b, label in turns)
    groups, refused = consolidate(prints, together_seconds(rows), **thresholds)
    labels = name_groups(groups, prints, (previous or {}).get('labels'))
    speakers = {}
    for label, ident in labels.items():
        entry = speakers.setdefault(ident, {'labels': [], 'seconds': 0.0, 'pitch': [], 'first': None})
        info = prints.get(label) or {}
        entry['labels'].append(label)
        entry['seconds'] += float(info.get('seconds') or 0.0)
        if info.get('pitch_hz'):
            entry['pitch'].append((float(info.get('seconds') or 0.0), float(info['pitch_hz'])))
        first = info.get('first')
        if first is not None and (entry['first'] is None or first < entry['first']):
            entry['first'] = float(first)
    for entry in speakers.values():
        pitch = sorted(entry.pop('pitch'), reverse=True)
        entry['labels'].sort()
        entry['seconds'] = round(entry['seconds'], 2)
        entry['pitch_hz'] = pitch[0][1] if pitch else None           # that of the label with the most speech
    return {'version': VERSION, 'covered_until': round(float(covered_until), 3), 'labels': labels, 'speakers': speakers,
            'turns': [list(row) for row in rows], 'refused': refused,
            'thresholds': {'merge_distance': thresholds.get('merge_distance', MERGE_DISTANCE),
                           'pitch_semitones': thresholds.get('pitch_semitones', PITCH_SEMITONES),
                           'together_limit': thresholds.get('together_limit', TOGETHER_SECONDS)}}


def save(path, artifact):
    """Atomic write; True when written (a failed write only delays the map, it never fails a window)."""
    path = Path(path)
    temporary = path.with_name(path.name + '.%d.tmp' % os.getpid())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(artifact, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
        os.replace(temporary, path)
        return True
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass
        return False


def load(path):
    """The artifact, or None when it is absent, unreadable or of another version."""
    try:
        artifact = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if (not isinstance(artifact, dict) or artifact.get('version') != VERSION or not isinstance(artifact.get('labels'), dict)
            or not isinstance(artifact.get('turns'), list)):
        return None
    return artifact


def state(artifact, window_end):
    """'ready' when the scout has analysed past the window's end, 'behind' when it has not yet, 'absent' without a map."""
    if not artifact:
        return 'absent'
    return 'ready' if float(artifact.get('covered_until') or 0.0) >= float(window_end) - 1e-6 else 'behind'


def map_window(live_turns, artifact, start, end, *, min_share=MAP_MIN_SHARE, min_seconds=MAP_MIN_SECONDS):
    """{live label: film id} for one window, by who speaks when.

    live_turns: the window's own turns ([{'start','end','speaker'}], film seconds); the artifact's scout turns carry the film
    ids. A live label gets the film speaker whose turns cover the largest part of its speech inside [start, end], when that
    part is at least ``min_seconds`` and ``min_share`` of the label's speech there. A label that fits no speaker that well is
    left out (it keeps its live label only)."""
    labels = (artifact or {}).get('labels') or {}
    scout = [(float(a), float(b), labels.get(str(label))) for a, b, label in (artifact or {}).get('turns') or ()
             if float(b) > start and float(a) < end and labels.get(str(label))]
    spoken, covered = {}, {}
    for turn in live_turns or ():
        a, b, label = max(float(turn['start']), start), min(float(turn['end']), end), str(turn['speaker'])
        if b <= a:
            continue
        spoken[label] = spoken.get(label, 0.0) + (b - a)
        for c, d, ident in scout:
            shared = min(b, d) - max(a, c)
            if shared > 0:
                covered.setdefault(label, {})
                covered[label][ident] = covered[label].get(ident, 0.0) + shared
    out = {}
    for label, total in spoken.items():
        options = covered.get(label) or {}
        if not options:
            continue
        ident, seconds = sorted(options.items(), key=lambda item: (-item[1], item[0]))[0]
        if seconds >= min_seconds and seconds >= min_share * total:
            out[label] = ident
    return out


def annotate(cues, window_map):
    """Set cue['film_speaker'] from the cue's live label (cue['local_speaker']); cues of unmapped labels are left alone.
    Returns how many cues were annotated."""
    count = 0
    for cue in cues or ():
        ident = (window_map or {}).get(str(cue.get('local_speaker')))
        if ident:
            cue['film_speaker'] = ident
            count += 1
    return count


def same_speaker(cue, donor):
    """Tighten-only rule for a borrowed reference: False when both lines have a film speaker and they differ. Unknown on
    either side is not a refusal (the existing rules decide alone, exactly as before the film map existed)."""
    a, b = (cue or {}).get('film_speaker'), (donor or {}).get('film_speaker')
    return not (a and b and a != b)
