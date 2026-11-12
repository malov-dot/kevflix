"""Reads the PC's server log and judges the TV control skins by the numbers the TV sent itself (tv_diagnostic lines with stage "controls",
relay/tv.py + tv/app/controls.js), the stop that must follow every leave, and the order of dubbing after a seek (tv_seek lines). Read-only.

    python scripts/controls_check.py [path to the server log]      (default: %LOCALAPPDATA%\\KevFLIX\\profile\\logs\\server.log, where
    F11 writes the event and request lines; server.stderr.log when there is no server.log, i.e. housekeeping.log_to_file false)

One line per skin session and per seek, then a summary; exit code 1 when anything FAILED."""
import datetime
import json
import os
import re
import sys

LINE = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d+) \w+ (.*)$')
PAINT_MEDIAN_MS, PAINT_WORST_MS, FRAME_MEDIAN_MS = 100, 250, 40
NEEDED_KEYS = ('MediaPlay', 'MediaPause', 'MediaPlayPause', 'MediaStop', 'MediaRewind', 'MediaFastForward')


def read(path):
    reports, seeks, ready, controls = {}, [], [], []
    with open(path, encoding='utf-8', errors='replace') as log:
        for line in log:
            match = LINE.match(line)
            if not match:
                continue
            at = datetime.datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S').timestamp() + int(match.group(2)) / 1000
            text = match.group(3)
            try:
                if text.startswith('tv_diagnostic '):
                    record = json.loads(text[len('tv_diagnostic '):])
                    if record.get('stage') == 'controls' and isinstance(record.get('controls'), dict):
                        reports[record.get('session')] = (at, match.group(1), record['controls'])        # the latest report of a session wins
                elif text.startswith('tv_seek '):
                    seeks.append((at, match.group(1), json.loads(text[len('tv_seek '):])))
                elif text.startswith('movie chunk_done '):
                    record = json.loads(text[len('movie chunk_done '):])
                    if record.get('state') in ('ready', 'approved'):
                        ready.append((at, record.get('job_id'), int(record.get('index'))))
                elif text.startswith('POST /tv-api/control'):
                    controls.append(at)
            except (ValueError, TypeError):
                continue
    return reports, seeks, ready, controls


def judge_report(at, c, controls):
    """[(name, verdict, detail)] for one session's report; verdict is PASS, FAIL or n/a."""
    out = []
    missing = [key for key in NEEDED_KEYS if key not in str(c.get('keys_ok') or '').split(',')]
    out.append(('media keys', 'FAIL' if missing else 'PASS', 'not registered: ' + ','.join(missing) if missing else 'failed (not needed): ' + (c.get('keys_failed') or 'none')))
    if (c.get('paint_n') or 0) >= 8:
        ok = c['paint_median_ms'] < PAINT_MEDIAN_MS and c['paint_worst_ms'] < PAINT_WORST_MS
        out.append(('key to paint', 'PASS' if ok else 'FAIL', 'median %s ms, worst %s ms over %s presses' % (c['paint_median_ms'], c['paint_worst_ms'], c['paint_n'])))
    else:
        out.append(('key to paint', 'n/a', 'fewer than 8 measured presses'))
    if (c.get('frame_n') or 0) >= 2:
        out.append(('frame time', 'PASS' if c['frame_median_ms'] <= FRAME_MEDIAN_MS else 'FAIL', 'median %s ms, worst %s ms, light look %s' % (c['frame_median_ms'], c.get('frame_worst_ms'), c.get('lite'))))
    else:
        out.append(('frame time', 'n/a', 'fewer than 2 bar showings measured'))
    presses, commits = c.get('seek_presses') or 0, c.get('seek_commits') or 0
    out.append(('seek collection', 'n/a' if not presses else 'PASS' if commits <= presses and (commits < presses or presses <= 2) else 'FAIL',
                '%s presses -> %s seeks, %s dropped' % (presses, commits, c.get('seek_dropped') or 0)))
    if c.get('rate_asked') is not None:
        out.append(('speed', 'PASS', 'asked %s, the TV plays %s%s' % (c['rate_asked'], c.get('rate_got'), '' if c.get('rate_got') == c['rate_asked'] else ' (refused: the menu shows the real speed)')))
    out.append(('skin errors', 'FAIL' if c.get('errors') else 'PASS', c.get('error_first') or 'none'))
    if c.get('when') == 'leave':
        stopped = any(at - 1 <= t <= at + 6 for t in controls)
        out.append(('stop after leave', 'PASS' if stopped else 'FAIL', 'a control request within 6 s of the leave' if stopped else 'no control request within 6 s of the leave'))
    return out


def judge_seek(at, seek, later, ready):
    """After a seek to window T the next two windows must be dubbed before any window of an abandoned position."""
    job, target = seek.get('job_id'), int(seek.get('index') or 0)
    order = [index for (t, j, index) in ready if j == job and at <= t < later]
    if target + 1 not in order or target + 2 not in order:
        return 'n/a', 'windows %d and %d were not both produced after this seek' % (target + 1, target + 2)
    before = order[:max(order.index(target + 1), order.index(target + 2))]
    stale = [index for index in before if index < target or index > target + 8]
    return ('FAIL', 'windows %s were dubbed first' % stale) if stale else ('PASS', 'order %s' % order[:6])


def main(path):
    reports, seeks, ready, controls = read(path)
    failed = 0
    for session, (at, stamp, c) in sorted(reports.items(), key=lambda item: item[1][0]):
        verdicts = judge_report(at, c, controls)
        failed += sum(1 for _, verdict, _ in verdicts if verdict == 'FAIL')
        print('%s session %s (%s, %s) codes %s repeat %s ended %s upnext %s' % (stamp, str(session)[:8], c.get('mode'), c.get('when'), c.get('codes'), c.get('repeat_seen'), c.get('ended'), c.get('upnext')))
        for name, verdict, detail in verdicts:
            print('    %-4s %-16s %s' % (verdict, name, detail))
    real = [s for s in seeks if s[2].get('action') == 'seek']
    for n, (at, stamp, seek) in enumerate(real):
        later = min([t for (t, _, other) in real[n + 1:] if other.get('job_id') == seek.get('job_id')] or [float('inf')])
        verdict, detail = judge_seek(at, seek, later, ready)
        failed += verdict == 'FAIL'
        print('%s seek to window %s (parked %s): %s %s' % (stamp, seek.get('index'), seek.get('parked'), verdict, detail))
    print('%d skin session(s), %d seek(s), %d FAILED' % (len(reports), len(real), failed))
    return 1 if failed else 0


if __name__ == '__main__':
    logs = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'KevFLIX', 'profile', 'logs')
    default = os.path.join(logs, 'server.log')                     # F11: the INFO lines go here (rotated)
    if not os.path.exists(default):
        default = os.path.join(logs, 'server.stderr.log')          # housekeeping.log_to_file false: everything on stderr
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else default))
