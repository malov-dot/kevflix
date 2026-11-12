#!/usr/bin/env python3
"""KevFLIX dub scorecard: one table per film from the stored window records. No model, no audio, CPU only, read-only.

    python scripts/dub_scorecard.py                      the film that was rendered last
    python scripts/dub_scorecard.py --job d939807e       one film by the start of its job id (repeat --job for more)
    python scripts/dub_scorecard.py --all                every film with at least --min-windows finished windows
    python scripts/dub_scorecard.py --expect film.json    also check film-specific bounds {"metric": {"min": x, "max": y}}
    python scripts/dub_scorecard.py --json out.json      also write every number as JSON

Exit code: 0 every target (and every --expect bound) is met for every film shown; 1 at least one is missed; 2 nothing to score.

The store is opened read-only and immutable, so this is safe while the app runs. Every number is computed from what the
pipeline already records for each 30 s window (chunks.payload). A number whose record is absent in an older build is shown
as "-" and is neither met nor missed. Standard library only.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sqlite3
import sys

VERSION = 'dub-scorecard-v1'

# name, comparison, bound. Rates are per cent. These hold for any film; film-specific numbers go into an --expect file.
# Each bound is what the fix specs of 2026-11-11 promise on the reference film, with a margin (see F18-scorecard.md).
TARGETS = (
    ('dubbed_lines_pct', '>=', 75.0),
    ('dubbed_speech_pct', '>=', 87.0),
    ('window_fallback', '<=', 0),
    ('kept_uncertain_pct', '<=', 9.5),
    ('kept_not_audible_pct', '<=', 1.5),
    ('within_03_pct', '>=', 78.0),
    ('early_05_pct', '<=', 12.0),
    ('late_015_pct', '<=', 11.0),
    ('hole_pct', '<=', 4.0),
    ('lift_over_4db', '<=', 0),
    ('level_off_1db_pct', '<=', 1.0),
    ('borrowed_pct', '<=', 6.0),
    ('language_flip_windows_pct', '<=', 2.0),
    ('foreign_unconfirmed_lines_pct', '<=', 1.0),
    ('untimed_words_dropped', '<=', 0),
    ('readback_mismatch_published_pct', '<=', 2.0),
    ('retry_pct', '<=', 12.0),
    ('render_ratio', '<=', 0.60),
    ('slow_windows_pct', '<=', 5.0),
)

# What each number means; printed with --explain and used as the row label.
LABELS = collections.OrderedDict((
    ('windows', 'windows scored (finished, with a record)'),
    ('film_seconds', 'film seconds covered by those windows'),
    ('lines', 'recognized lines'),
    ('dubbed_lines', 'lines dubbed'),
    ('dubbed_lines_pct', 'lines dubbed, % of lines'),
    ('dubbed_speech_pct', 'speech time dubbed, % of line seconds'),
    ('short_lines', 'lines under 1 s'),
    ('short_lines_dubbed', 'lines under 1 s dubbed'),
    ('window_fallback', 'whole windows that fell back to the original (original-trial-fallback)'),
    ('no_dialogue_windows', 'windows with no dialogue'),
    ('original_windows', 'windows with every line kept original'),
    ('speech_not_transcribed_windows', 'windows kept original with speech nobody transcribed'),
    ('kept_no_reference', 'kept original: no own voice reference'),
    ('kept_no_reference_pct', 'kept original: no own voice reference, % of lines'),
    ('kept_uncertain', 'kept original: transcript not usable'),
    ('kept_uncertain_pct', 'kept original: transcript not usable, % of lines'),
    ('kept_hallucination', '  of which invented text (credits, tags, foreign script, empty)'),
    ('kept_english', '  of which English (text, audio, already English)'),
    ('kept_nonword', '  of which non-word sounds (interjection, filler, burst, fragment)'),
    ('kept_singing', '  of which singing'),
    ('kept_chant', '  of which chant rule'),
    ('kept_quiet', '  of which no signal / inaudible / sparse'),
    ('kept_low_confidence', '  of which low recognizer confidence only'),
    ('kept_translator', '  of which translator could not'),
    ('kept_not_audible', 'kept original: take not audible in the mix'),
    ('kept_not_audible_pct', 'kept original: take not audible in the mix, % of lines'),
    ('kept_readback', 'kept original: take read back with a critical change'),
    ('kept_overrun', 'kept original: take did not fit its slot'),
    ('kept_other', 'kept original: other'),
    ('takes', 'dubbed takes with a placement'),
    ('within_03_pct', 'takes ending within 0.3 s of the line end, %'),
    ('early_05_pct', 'takes ending more than 0.5 s before the line end, %'),
    ('late_015_pct', 'takes ending more than 0.15 s after the line end, %'),
    ('late_03_pct', 'takes ending more than 0.3 s after the line end, %'),
    ('start_late_02', 'takes starting more than 0.2 s late'),
    ('hole_seconds', 'seconds without a voice where the line was removed (take ended > 0.3 s early)'),
    ('hole_pct', 'the same, % of dubbed line seconds'),
    ('sped_up_takes', 'takes sped up'),
    ('slowed_takes', 'takes slowed down'),
    ('timed_asked', 'lines asked for a timed wording (before the take)'),
    ('timed_adopted', 'timed wordings used'),
    ('measured_retranslations', 'lines re-translated from the measured take'),
    ('measured_retranslation_seconds', 'seconds spent on them'),
    ('measured_retranslation_skipped', 'targets skipped by the time budget'),
    ('level_off_1db_pct', 'takes more than 1 dB off the level of their original line (the lift apart), %'),
    ('lift_over_4db', 'takes lifted by more than 4 dB'),
    ('lifted_takes', 'takes lifted to be heard (2 or 4 dB)'),
    ('bed_ducked_takes', 'takes with the bed lowered under them'),
    ('weak_takes_pct', 'takes less than 3 dB over the rest of the mix, %'),
    ('ref_own', 'lines cloned from their own audio (own, trimmed, extended)'),
    ('borrowed', 'lines given another line of the same voice as reference'),
    ('borrowed_dubbed', 'of them dubbed'),
    ('borrowed_pct', 'dubbed with a borrowed reference, % of dubbed lines'),
    ('ref_refused', 'own references refused'),
    ('two_label_lines_pct', 'lines of 1 s or more with two speaker labels on >= 20 % of their words, %'),
    ('speaker_labels', 'speaker labels in the film'),
    ('speakers_per_window_median', 'speaker labels per window, median'),
    ('film_language', 'film language'),
    ('language_flip_windows', 'windows decoded in another language than the film'),
    ('language_flip_windows_pct', 'the same, % of windows'),
    ('foreign_unconfirmed_lines', 'lines written in another language with no audio confirmation'),
    ('foreign_unconfirmed_lines_pct', 'the same, % of lines'),
    ('english_audio_lines', 'lines confirmed English by their audio (kept original)'),
    ('film_lock_guard_readings', 'second readings in the window language (film lock guard)'),
    ('language_corrections', 'segments decoded again in their own language'),
    ('untimed_words_dropped', 'recognized words dropped for having no duration'),
    ('untimed_words_kept', 'such words kept with their neighbour'),
    ('untranscribed_stretches', 'stretches of detector speech (1 s or more) that no recognized word covers'),
    ('untranscribed_speech_seconds', 'their length, seconds'),
    ('rescue_reads', 'extra reads of missed speech'),
    ('rescued_segments', 'segments they added'),
    ('second_readings', 'second readings of weak segments'),
    ('readback_mismatch_published', 'published takes read back as other words (word error >= 1.0)'),
    ('readback_mismatch_published_pct', 'the same, % of dubbed lines'),
    ('render_seconds', 'render seconds (all windows)'),
    ('render_ratio', 'render seconds per film second'),
    ('slow_windows_pct', 'windows that took longer than they play, %'),
    ('window_seconds_p90', 'render seconds per window, 90th percentile'),
    ('attempts_3plus_windows', 'windows rendered 3 or more times'),
    ('attempts_3plus_windows_pct', 'the same, % of windows'),
    ('retry_seconds', 'seconds of repeated render attempts'),
    ('retry_pct', 'the same, % of render seconds'),
    ('vox_two_instances_pct', 'windows that ran two voice instances, % of windows with takes'),
    ('stage_asr', 'stage seconds: recognition'),
    ('stage_speakers', 'stage seconds: speakers'),
    ('stage_translation', 'stage seconds: translation'),
    ('stage_cloning', 'stage seconds: cloning'),
    ('stage_mix', 'stage seconds: scheduling and mix'),
))

SECTIONS = (
    ('Coverage', ('windows', 'film_seconds', 'lines', 'dubbed_lines', 'dubbed_lines_pct', 'dubbed_speech_pct', 'short_lines',
                  'short_lines_dubbed', 'window_fallback', 'no_dialogue_windows', 'original_windows', 'speech_not_transcribed_windows')),
    ('Why lines stayed original', ('kept_no_reference', 'kept_no_reference_pct', 'kept_uncertain', 'kept_uncertain_pct', 'kept_hallucination',
                                   'kept_english', 'kept_nonword', 'kept_singing', 'kept_chant', 'kept_quiet', 'kept_low_confidence',
                                   'kept_translator', 'kept_not_audible', 'kept_not_audible_pct', 'kept_readback', 'kept_overrun', 'kept_other')),
    ('Timing', ('takes', 'within_03_pct', 'early_05_pct', 'late_015_pct', 'late_03_pct', 'start_late_02', 'hole_seconds', 'hole_pct',
                'sped_up_takes', 'slowed_takes', 'timed_asked', 'timed_adopted', 'measured_retranslations',
                'measured_retranslation_seconds', 'measured_retranslation_skipped')),
    ('Loudness', ('level_off_1db_pct', 'lift_over_4db', 'lifted_takes', 'bed_ducked_takes', 'weak_takes_pct')),
    ('Voices', ('ref_own', 'borrowed', 'borrowed_dubbed', 'borrowed_pct', 'ref_refused', 'two_label_lines_pct', 'speaker_labels', 'speakers_per_window_median')),
    ('Transcript', ('film_language', 'language_flip_windows', 'language_flip_windows_pct', 'foreign_unconfirmed_lines',
                    'foreign_unconfirmed_lines_pct', 'english_audio_lines', 'film_lock_guard_readings', 'language_corrections',
                    'untimed_words_dropped', 'untimed_words_kept', 'untranscribed_stretches', 'untranscribed_speech_seconds', 'rescue_reads', 'rescued_segments',
                    'second_readings', 'readback_mismatch_published', 'readback_mismatch_published_pct')),
    ('Speed', ('render_seconds', 'render_ratio', 'slow_windows_pct', 'window_seconds_p90', 'attempts_3plus_windows',
               'attempts_3plus_windows_pct', 'retry_seconds', 'retry_pct', 'vox_two_instances_pct', 'stage_asr', 'stage_speakers', 'stage_translation',
               'stage_cloning', 'stage_mix')),
)

HALLUCINATION = {'credits_hallucination', 'sound_tag', 'foreign_script', 'no_text'}
ENGLISH = {'english_text', 'english_audio', 'already_english'}
NONWORD = {'interjection', 'filler', 'vocalization', 'scream', 'fragment', 'laugh', 'unvoiced_burst', 'loud_vowel'}
SINGING = {'singing_label', 'singing_held_note'}
CHANT = {'chant_monotone', 'chant_recitation'}
QUIET = {'no_signal', 'inaudible', 'sparse_speech', 'too_short'}
TRANSLATOR = {'cyrillic_in_english', 'transliteration_not_translation', 'translation_truncated', 'translation_empty',
              'translation_input_too_long', 'translation_input_empty', 'source_language_unresolved'}


def default_db():
    return os.path.join(os.environ.get('LOCALAPPDATA') or os.path.expanduser('~'), 'KevFLIX', 'profile', 'state', 'relay.sqlite3')


def connect(path):
    return sqlite3.connect('file:' + os.path.abspath(path).replace(os.sep, '/') + '?mode=ro&immutable=1', uri=True)


def number(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if value == value and abs(value) != float('inf') else default


def pct(part, whole):
    return round(100.0 * part / whole, 1) if whole else None


def percentile(values, share):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * share))] if values else None


def median(values):
    values = sorted(values)
    if not values:
        return None
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0


def untranscribed_stretches(report, minimum=1.0, pad=.2, context=3.0):
    """[(start, end)] absolute: stretches of at least ``minimum`` seconds of the detector's speaking windows (raw_turns, any
    speaker) that no recognized word (probability >= .25, widened by ``pad``) covers and that START inside the window's own
    render range. The same measure as relay.asr_rescue.speech_gap_windows before its limits (F16)."""
    try:
        low, high, origin = float(report['render_start']), float(report['render_end']), float(report.get('origin') or 0.0)
    except (KeyError, TypeError, ValueError):
        return []
    end = number((report.get('progressive_timing') or {}).get('end'), high + context)
    recognized = []
    for segment in report.get('transcript') or ():
        for word in segment.get('words') or [segment]:
            if number(word.get('probability', 1), 1.0) >= .25:
                recognized.append((origin + number(word.get('start')) - pad, origin + number(word.get('end')) + pad))
    recognized.sort()
    speaking, right = [], None
    for a, b in sorted((max(origin, number(t.get('start'))), min(end, number(t.get('end')))) for t in report.get('raw_turns') or ()):
        if b <= a:
            continue
        if right is None or a > right:
            speaking.append([a, b])
            right = b
        elif b > right:
            speaking[-1][1] = b
            right = b
    gaps = []
    for a, b in speaking:
        cursor = a
        for c, d in recognized:
            if d <= cursor or c >= b:
                continue
            if c > cursor:
                gaps.append((cursor, min(c, b)))
            cursor = max(cursor, d)
            if cursor >= b:
                break
        if cursor < b:
            gaps.append((cursor, b))
    return [(a, b) for a, b in gaps if b - a >= minimum and low <= a < high]


def two_labels(cue, turns):
    share, total = collections.defaultdict(float), 0.0
    for word in cue.get('words') or ():
        try:
            a, b = float(word['start']), float(word['end'])
        except (KeyError, TypeError, ValueError):
            continue
        total += b - a
        for turn in turns:
            share[turn.get('speaker')] += max(0.0, min(b, number(turn.get('end'))) - max(a, number(turn.get('start'))))
    return total > 0 and sum(1 for value in share.values() if value / total >= .2) >= 2


def kept_class(cue, reasons, issue_reason, runtime):
    """One class for a line that was not dubbed."""
    runtime = ' '.join(runtime or ())
    if 'take_not_audible_in_render' in runtime:
        return 'kept_not_audible'
    if 'critical_readback_mismatch' in runtime:
        return 'kept_readback'
    if 'speech_timing_overrun' in runtime:
        return 'kept_overrun'
    if 'actor_reference_unavailable' in reasons:
        return 'kept_no_reference'
    if 'uncertain_transcript' in reasons:
        return 'kept_uncertain'
    return 'kept_other'


def uncertain_class(cue, issue_reason):
    reason = str(cue.get('nonverbal') or issue_reason or '')
    head = reason.split(':')[0]
    if head in HALLUCINATION:
        return 'kept_hallucination'
    if head in ENGLISH:
        return 'kept_english'
    if head in SINGING:
        return 'kept_singing'
    if head in CHANT:
        return 'kept_chant'
    if head in QUIET:
        return 'kept_quiet'
    if head in TRANSLATOR or head.startswith('translation_'):
        return 'kept_translator'
    if head in NONWORD or head.startswith('nonspeech_burst'):
        return 'kept_nonword'
    return 'kept_low_confidence' if not reason else 'kept_nonword'


def score(job, windows):
    """All numbers of one film. job: the job payload (dict); windows: [(idx, payload dict)] of finished windows."""
    n = collections.Counter()
    m = {}
    segment = number(job.get('segment_seconds'), 30.0) or 30.0
    end_errors, start_delays, elapsed, deviations, lifts, weak, speakers_per_window = [], [], [], [], [], [], []
    languages, labels = collections.Counter(), set()
    hole = dubbed_seconds = line_seconds = 0.0
    stages = collections.Counter()
    retry_seconds, retry_known = 0.0, False
    two_label, long_lines = 0, 0
    vox_windows = vox_two = 0
    per_window = []
    for idx, p in windows:
        mode = p.get('mode')
        n['windows'] += 1
        cues = p.get('cues') or []
        published = set(p.get('published_cue_ids') or [])
        language = (p.get('language') or {})
        detected = language.get('detected') if language.get('basis') == 'film-lock' else language
        per_window.append((idx, p, cues, (detected or {}).get('language'), language.get('language')))
        if cues and (detected or {}).get('language'):
            languages[(detected or {}).get('language')] += 1
        n['window_fallback'] += mode == 'original-trial-fallback'
        n['no_dialogue_windows'] += mode == 'no-dialogue'
        reason = p.get('beta_original_reason') or []
        n['original_windows'] += mode == 'original-unverified' and 'all_selected_cues_original' in reason
        n['speech_not_transcribed_windows'] += mode == 'original-unverified' and 'speech_not_transcribed' in reason
        fallback = p.get('beta_original_fallback') or {}
        runtime = p.get('beta_runtime_fallback') or {}
        issue_reason = {i.get('cue'): i.get('reason') for i in p.get('beta_cue_fallback_issues') or [] if isinstance(i, dict)}
        turns = p.get('raw_turns') or []
        for turn in turns:
            labels.add(turn.get('speaker'))
        speakers_per_window.append(len({t.get('speaker') for t in turns}))
        for cue in cues:
            seconds = number(cue.get('end')) - number(cue.get('start'))
            n['lines'] += 1
            line_seconds += seconds
            n['short_lines'] += seconds < 1.0
            if seconds >= 1.0 and cue.get('words'):
                long_lines += 1
                two_label += two_labels(cue, turns)
            if cue.get('id') in published:
                n['dubbed_lines'] += 1
                dubbed_seconds += seconds
                n['short_lines_dubbed'] += seconds < 1.0
                continue
            kind = kept_class(cue, fallback.get(cue.get('id')) or (), issue_reason.get(cue.get('id')), runtime.get(cue.get('id')))
            n[kind] += 1
            if kind == 'kept_uncertain':
                n[uncertain_class(cue, issue_reason.get(cue.get('id')))] += 1
        outcomes = ((p.get('line_reference_outcomes') or {}).get('cues') or {})
        for cid, outcome in outcomes.items():
            kind = outcome.get('kind')
            n['ref_own'] += kind in ('own', 'own-trimmed', 'extended')
            n['borrowed'] += kind == 'borrowed'
            n['borrowed_dubbed'] += kind == 'borrowed' and cid in published
            n['ref_refused'] += bool(outcome.get('refused_reason'))
        utterances = {u.get('cue'): u for u in p.get('utterances') or [] if isinstance(u, dict)}
        for item in p.get('published_utterances') or []:
            if item.get('kind') != 'generated':
                continue
            place = item.get('placement') or {}
            if 'source_end' not in place or 'end' not in place:
                continue
            n['takes'] += 1
            error = number(place.get('end')) - number(place.get('source_end'))
            end_errors.append(error)
            start_delays.append(number(place.get('start')) - number(place.get('source_start')))
            hole += max(0.0, -error - .3)
            take = utterances.get(item.get('cue_id')) or {}
            levels = take.get('mix_levels') or {}
            lift = number(levels.get('audibility_boost_db'))
            lifts.append(lift)
            if 'level_deviation_db' in levels:
                deviations.append(abs(number(levels.get('level_deviation_db')) - lift))      # the permitted lift is not a mismatch
            n['bed_ducked_takes'] += number(levels.get('bed_duck_db')) > 0
            n['sped_up_takes'] += bool((take.get('fit') or {}).get('applied'))
            n['slowed_takes'] += bool((take.get('slot_fill') or {}).get('applied'))
            heard = (item.get('rendered_audibility') or {}).get('take_to_rest_db_median')
            if isinstance(heard, (int, float)):
                weak.append(heard < 3.0)
            if number(take.get('readback_word_error')) >= 1.0:
                n['readback_mismatch_published'] += 1
        for row in p.get('discarded') or []:
            if isinstance(row, dict) and row.get('reason') == 'asr_invalid_interval':
                n['untimed_words_kept' if row.get('text_kept_with') else 'untimed_words_dropped'] += 1
        for row in p.get('language_corrections') or []:
            n['language_corrections'] += 1
            n['film_lock_guard_readings'] += isinstance(row, dict) and row.get('basis') == 'film-lock-guard'
        for row in p.get('asr_rescue') or []:
            n['rescue_reads'] += 1
            n['rescued_segments'] += int(number((row or {}).get('added_segments')))
        n['second_readings'] += len(p.get('asr_second_readings') or [])
        timed = p.get('timed_translation') or {}
        n['timed_asked'] += len(timed)
        n['timed_adopted'] += sum(1 for row in timed.values() if isinstance(row, dict) and row.get('adopted'))
        measured = p.get('measured_retranslation') or {}
        n['measured_retranslations'] += int(number(measured.get('lines')))
        n['measured_retranslation_skipped'] += int(number(measured.get('skipped')))
        m['measured_retranslation_seconds'] = round(m.get('measured_retranslation_seconds', 0.0) + number(measured.get('seconds')), 1)
        stretches = untranscribed_stretches(p)
        n['untranscribed_stretches'] += len(stretches)
        m['untranscribed_speech_seconds'] = round(m.get('untranscribed_speech_seconds', 0.0) + sum(b - a for a, b in stretches), 1)
        if p.get('elapsed_seconds') is not None:
            elapsed.append(number(p.get('elapsed_seconds')))
        ledger = p.get('beta_line_ledger') or {}
        n['attempts_3plus_windows'] += number(ledger.get('attempt')) >= 3
        if 'earlier_attempt_seconds' in ledger:
            retry_known = True
            retry_seconds += sum(number(x) for x in ledger.get('earlier_attempt_seconds') or [])
        for key, value in (p.get('timing') or {}).items():
            stages[key] += number(value)
        vox = p.get('vox_concurrency') or {}
        if number(vox.get('takes')) > 0:
            vox_windows += 1
            vox_two += number(vox.get('bound')) >= 2
    film = languages.most_common(1)[0][0] if languages else None
    for idx, p, cues, detected, decoded in per_window:
        used = decoded or detected
        if film and used and used != film:
            n['language_flip_windows'] += 1
        for cue in cues:
            if film and cue.get('language') and cue.get('language') != film and not cue.get('language_from_audio'):
                n['foreign_unconfirmed_lines'] += 1
            if film and film != 'en' and cue.get('language') == 'en' and cue.get('language_from_audio'):
                n['english_audio_lines'] += 1
    for key in LABELS:
        if key in n or key in ('window_fallback', 'untimed_words_dropped', 'attempts_3plus_windows', 'language_flip_windows',
                               'foreign_unconfirmed_lines', 'kept_not_audible', 'kept_no_reference', 'kept_uncertain',
                               'readback_mismatch_published', 'borrowed', 'borrowed_dubbed', 'dubbed_lines', 'lines', 'windows',
                               'english_audio_lines', 'untranscribed_stretches', 'bed_ducked_takes', 'timed_asked', 'timed_adopted',
                               'measured_retranslations', 'measured_retranslation_skipped', 'film_lock_guard_readings',
                               'rescue_reads', 'rescued_segments', 'second_readings', 'untimed_words_kept', 'speech_not_transcribed_windows',
                               'kept_readback', 'kept_overrun', 'kept_other', 'kept_hallucination', 'kept_english', 'kept_nonword',
                               'kept_singing', 'kept_chant', 'kept_quiet', 'kept_low_confidence', 'kept_translator', 'short_lines',
                               'short_lines_dubbed', 'no_dialogue_windows', 'original_windows', 'ref_own', 'ref_refused',
                               'sped_up_takes', 'slowed_takes', 'language_corrections', 'takes'):
            m[key] = n[key]
    takes, lines, dubbed, count = n['takes'], n['lines'], n['dubbed_lines'], n['windows']
    m['film_seconds'] = round(count * segment, 1)
    m['film_language'] = film
    m['dubbed_lines_pct'] = pct(dubbed, lines)
    m['dubbed_speech_pct'] = pct(dubbed_seconds, line_seconds)
    for key in ('kept_no_reference', 'kept_uncertain', 'kept_not_audible', 'foreign_unconfirmed_lines'):
        m[key + '_pct'] = pct(n[key], lines)
    m['within_03_pct'] = pct(sum(abs(e) <= .3 for e in end_errors), takes)
    m['early_05_pct'] = pct(sum(e < -.5 for e in end_errors), takes)
    m['late_015_pct'] = pct(sum(e > .15 for e in end_errors), takes)
    m['late_03_pct'] = pct(sum(e > .3 for e in end_errors), takes)
    m['start_late_02'] = sum(d > .2 for d in start_delays)
    m['hole_seconds'] = round(hole, 1)
    m['hole_pct'] = pct(hole, dubbed_seconds)
    m['level_off_1db_pct'] = pct(sum(d > 1.0 for d in deviations), len(deviations)) if deviations else None
    m['lift_over_4db'] = sum(l > 4.0 + 1e-6 for l in lifts)
    m['lifted_takes'] = sum(l > 0 for l in lifts)
    m['weak_takes_pct'] = pct(sum(weak), len(weak))
    m['borrowed_pct'] = pct(n['borrowed_dubbed'], dubbed)
    m['two_label_lines_pct'] = pct(two_label, long_lines)
    m['speaker_labels'] = len(labels - {None})
    m['speakers_per_window_median'] = median(speakers_per_window)
    m['language_flip_windows_pct'] = pct(n['language_flip_windows'], count)
    m['readback_mismatch_published_pct'] = pct(n['readback_mismatch_published'], dubbed)
    m['render_seconds'] = round(sum(elapsed), 1)
    m['render_ratio'] = round(sum(elapsed) / (count * segment), 3) if count else None
    m['slow_windows_pct'] = pct(sum(e > segment for e in elapsed), len(elapsed))
    m['window_seconds_p90'] = round(percentile(elapsed, .9), 1) if elapsed else None
    m['attempts_3plus_windows_pct'] = pct(n['attempts_3plus_windows'], count)
    m['retry_seconds'] = round(retry_seconds, 1) if retry_known else None
    m['retry_pct'] = pct(retry_seconds, sum(elapsed)) if retry_known else None
    m['vox_two_instances_pct'] = pct(vox_two, vox_windows)
    for key, stage in (('stage_asr', 'asr'), ('stage_speakers', 'speakers'), ('stage_translation', 'translation'),
                       ('stage_cloning', 'cloning'), ('stage_mix', 'scheduling_and_mix')):
        m[key] = round(stages.get(stage, 0.0), 1)
    return m


def verdict(value, comparison, bound):
    if value is None:
        return None
    return value >= bound if comparison == '>=' else value <= bound


def check(metrics, expect=None):
    """[(name, comparison, bound, value, met, source)] for the general targets and the film-specific --expect bounds."""
    rows = [(name, comparison, bound, metrics.get(name), verdict(metrics.get(name), comparison, bound), 'target')
            for name, comparison, bound in TARGETS]
    for name, bounds in sorted((expect or {}).items()):
        if not isinstance(bounds, dict):
            continue
        for key, comparison in (('min', '>='), ('max', '<=')):
            if key in bounds:
                value = metrics.get(name)
                value = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
                rows.append((name, comparison, bounds[key], value, verdict(value, comparison, bounds[key]), 'expect'))
    return rows


def show(value):
    if value is None:
        return '-'
    if isinstance(value, float):
        return ('%.3f' % value).rstrip('0').rstrip('.') if abs(value) < 10 else ('%.1f' % value).rstrip('0').rstrip('.')
    return str(value)


def table(title, metrics, checks):
    by_name = collections.defaultdict(list)
    for name, comparison, bound, value, met, source in checks:
        by_name[name].append((comparison, bound, met, source))
    lines = ['', '=' * 132, title, '=' * 132]
    for section, names in SECTIONS:
        lines.append('')
        lines.append('%-34s %9s  %-11s %-12s %s' % (section.upper(), 'value', 'target', '', ''))
        lines.append('-' * 132)
        for name in names:
            if name not in metrics:
                continue
            bounds = by_name.get(name) or [(None, None, None, None)]
            for position, (comparison, bound, met, source) in enumerate(bounds):
                target = '' if comparison is None else '%s %s%s' % (comparison, show(bound), ' *' if source == 'expect' else '')
                state = '' if comparison is None else 'not recorded' if met is None else 'ok' if met else 'MISSED'
                lines.append('%-34s %9s  %-11s %-12s %s' % (name if position == 0 else '', show(metrics[name]) if position == 0 else '',
                                                           target, state, LABELS[name].strip() if position == 0 else ''))
    missed = [(name, comparison, bound, value) for name, comparison, bound, value, met, source in checks if met is False]
    unknown = [name for name, comparison, bound, value, met, source in checks if met is None]
    lines.append('')
    lines.append('RESULT: %s' % ('all targets met' if not missed else '%d missed: %s' % (
        len(missed), ', '.join('%s %s (%s %s)' % (name, show(value), comparison, show(bound)) for name, comparison, bound, value in missed))))
    if unknown:
        lines.append('not recorded by the build that rendered this film (neither met nor missed): %s' % ', '.join(unknown))
    lines.append('(* = film-specific bound from --expect)')
    return '\n'.join(lines), bool(missed)


def films(con, prefixes, everything, minimum):
    jobs = con.execute('select id, payload, updated from jobs order by updated').fetchall()
    counts = dict(con.execute("select job_id, count(*) from chunks where state='ready' group by job_id").fetchall())
    if prefixes:
        chosen = [row for row in jobs if any(row[0].startswith(prefix) for prefix in prefixes)]
    else:
        chosen = [row for row in jobs if counts.get(row[0], 0) >= minimum]
        if not everything:
            chosen = chosen[-1:]
    for job_id, payload, updated in chosen:
        try:
            job = json.loads(payload or '{}')
        except ValueError:
            job = {}
        windows = []
        for idx, body in con.execute("select idx, payload from chunks where job_id=? and state='ready' order by idx", (job_id,)):
            try:
                record = json.loads(body or '{}')
            except ValueError:
                continue
            if isinstance(record, dict) and record.get('mode'):
                windows.append((idx, record))
        yield job_id, job, windows


def main(argv=None):
    parser = argparse.ArgumentParser(description='KevFLIX dub scorecard (no model, read-only)')
    parser.add_argument('--db', default=default_db(), help='relay.sqlite3 (default: the installed app\'s store)')
    parser.add_argument('--job', action='append', default=[], help='start of a job id; repeat for more films')
    parser.add_argument('--all', action='store_true', help='every film with at least --min-windows finished windows')
    parser.add_argument('--min-windows', type=int, default=20)
    parser.add_argument('--expect', help='JSON file {"metric": {"min": x, "max": y}}: film-specific bounds, checked as well')
    parser.add_argument('--json', help='write every number of every film shown to this file')
    parser.add_argument('--explain', action='store_true', help='print what each number means and exit')
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    if args.explain:
        for name, label in LABELS.items():
            print('%-34s %s' % (name, label.strip()))
        return 0
    if not os.path.isfile(args.db):
        print('store not found: %s' % args.db)
        return 2
    expect = None
    if args.expect:
        with open(args.expect, encoding='utf-8') as handle:
            expect = json.load(handle)
    con = connect(args.db)
    missed_any, shown, out = False, 0, {}
    try:
        for job_id, job, windows in films(con, args.job, args.all, args.min_windows):
            if not windows:
                print('job %s: no finished window with a record' % job_id[:8])
                continue
            metrics = score(job, windows)
            checks = check(metrics, expect)
            title = '%s   job %s   %d windows   (%s)' % (str(job.get('title') or 'untitled')[:60], job_id[:8], len(windows), VERSION)
            text, missed = table(title, metrics, checks)
            print(text)
            missed_any |= missed
            shown += 1
            out[job_id] = {'title': job.get('title'), 'metrics': metrics,
                           'missed': [name for name, comparison, bound, value, met, source in checks if met is False]}
    finally:
        con.close()
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as handle:
            json.dump({'version': VERSION, 'films': out}, handle, ensure_ascii=False, indent=1)
    if not shown:
        print('nothing to score')
        return 2
    return 1 if missed_any else 0


if __name__ == '__main__':
    sys.exit(main())
