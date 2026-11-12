"""Film language lock (F02, 2026-11-11): the language a film is spoken in is decided once, from its confident windows.

Every 30 s window used to be decoded in whatever language the recognizer detected on that window alone. On a window with
little speech, music, or an English scene the detection flips (a full Russian film: 41 of 261 windows were not 'ru'), the whole
window is then written down in the wrong language, and Russian speech decoded as English words is kept original as "already
English". The film's language does not change every 30 seconds.

How it works
  * Each window casts ONE vote: the language detected on it, when the detection is confident (probability >= 0.90) and the
    window really has speech (at least 2 recognized segments with a mean word probability of 0.5 or more). Votes are kept in
    <job folder>/film-language.json, keyed by the window, so preparing a window again replaces its vote.
  * The film is LOCKED to a language once it has at least 5 votes and at least 80 % of all votes. A film that really is in two
    languages never reaches 80 % and keeps the per-window detection.
  * A locked film is decoded in that language. Real speech in another language inside it (an English scene) is still found per
    segment from its audio and decoded in that language (actor_pipeline._correct_languages).
Pure functions plus one small JSON file; no model, no store schema change. Standard library only.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

VERSION = 'film-language-v1'
FILE_NAME = 'film-language.json'
VOTE_MIN_PROBABILITY = .90      # a window votes only with a detection at least this sure
VOTE_MIN_SEGMENTS = 2           # ... and at least this many clearly recognized segments (a window of music does not vote)
VOTE_SEGMENT_CONFIDENCE = .5    # mean word probability that makes a segment count
LOCK_MIN_VOTES = 5
LOCK_MIN_SHARE = .80
LOCKED_PROBABILITY = 1.0        # what report['language']['probability'] carries for a locked film (a decision, not a detection)


def enabled(beta):
    """actor_beta.film_language_lock (default on with the beta lane; only an explicit false turns it off)."""
    return isinstance(beta, dict) and beta.get('enabled') is True and beta.get('film_language_lock') is not False


def path_for(job_folder):
    return Path(job_folder) / FILE_NAME


def empty():
    return {'version': VERSION, 'votes': {}}


def load(path):
    """The stored votes; an absent, unreadable or foreign file is an empty state (the film is then simply not locked yet)."""
    try:
        state = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return empty()
    if not isinstance(state, dict) or state.get('version') != VERSION or not isinstance(state.get('votes'), dict):
        return empty()
    return {'version': VERSION, 'votes': {str(k): v for k, v in state['votes'].items() if isinstance(v, dict)}}


def clear_segments(segments):
    """How many recognizer segments carry clearly recognized words (mean word probability >= VOTE_SEGMENT_CONFIDENCE)."""
    count = 0
    for segment in segments or ():
        values = [float(word.get('probability', 0.)) for word in segment.get('words') or ()]
        if values and sum(values) / len(values) >= VOTE_SEGMENT_CONFIDENCE:
            count += 1
    return count


def vote_of(detected, segments):
    """The vote of one window ({'language', 'probability', 'segments'}) or None when it may not vote."""
    if not isinstance(detected, dict):
        return None
    language, probability = detected.get('language'), detected.get('probability')
    if not isinstance(language, str) or not language or language in ('und', 'unknown', 'auto'):
        return None
    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not math.isfinite(probability):
        return None
    count = clear_segments(segments)
    if probability < VOTE_MIN_PROBABILITY or count < VOTE_MIN_SEGMENTS:
        return None
    return {'language': language, 'probability': round(float(probability), 4), 'segments': count}


def cast(state, window_key, detected, segments):
    """A NEW state with this window's vote set (or removed when it may not vote). Pure."""
    votes = dict(state.get('votes') or {})
    vote = vote_of(detected, segments)
    if vote is None:
        votes.pop(str(window_key), None)
    else:
        votes[str(window_key)] = vote
    return {'version': VERSION, 'votes': votes}


def decide(state):
    """{'language', 'votes', 'of', 'share'} when the film is locked, else None."""
    votes = [v.get('language') for v in (state.get('votes') or {}).values() if isinstance(v, dict) and v.get('language')]
    if len(votes) < LOCK_MIN_VOTES:
        return None
    counts = {}
    for language in votes:
        counts[language] = counts.get(language, 0) + 1
    language, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    share = count / len(votes)
    if count < LOCK_MIN_VOTES or share < LOCK_MIN_SHARE:
        return None
    return {'language': language, 'votes': count, 'of': len(votes), 'share': round(share, 4)}


def save(path, state):
    """Write the state atomically; True when written. A failed write (folder gone, file held open by another process) only
    loses this window's vote, so it is swallowed: the lock is evidence that accumulates, never a reason to fail a window."""
    path = Path(path)
    temporary = path.with_name(path.name + '.%d.tmp' % os.getpid())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(state, ensure_ascii=False, sort_keys=True), encoding='utf-8')
        os.replace(temporary, path)
        return True
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass
        return False


def window_key(render_start):
    return '%.3f' % float(render_start)


def language_record(lock, detected):
    """report['language'] for a window of a locked film: the film's language, with the window's own detection kept as evidence."""
    return {'language': lock['language'], 'probability': LOCKED_PROBABILITY, 'basis': 'film-lock', 'film': dict(lock),
            'detected': dict(detected) if isinstance(detected, dict) else None}
