"""Place natural-duration actor lines into verified dialogue gaps."""
from __future__ import annotations
import math

OVERRUN_TOLERANCE_SECONDS=.0001   # the tolerance of the overrun decision below (take_fit.OVERRUN_TOLERANCE_SECONDS)
_FIT_MATCH_SECONDS=.0005          # a fit receipt must describe THIS generated duration, to half a millisecond


def schedule_lines(cues,durations,*,origin,end,max_early_seconds=.75,gap_seconds=.04,fits=None):
    """Place each generated line; a line that cannot fit before the next wall is reported, never moved across it.

    ``fits`` (optional, default None = exactly the previous behaviour): a sequence parallel to ``durations`` whose
    entries are None or a relay.take_fit.fit_take receipt. ``durations`` stay the GENERATED (unfitted) lengths; for
    an entry with a receipt the scheduler uses the receipt's fitted ``after_seconds`` for the overrun decision, so a
    take that the single atempo fit brings inside its slot is no longer an overrun, and one that still does not fit
    stays a ``speech_timing_overrun`` carrying the fit facts. Rows of fitted lines report the applied tempo in
    ``speed`` (the audio is ALREADY fitted; ``speed`` is a record, not an instruction to stretch again)."""
    if len(cues)!=len(durations):raise ValueError('Each line needs its generated duration')
    if not 0<=max_early_seconds<=2 or gap_seconds<0:raise ValueError('Invalid dialogue scheduling limits')
    scheduled=durations if fits is None else fitted_durations(durations,fits)
    result,issues=[],[]
    previous=origin
    for index,(cue,duration) in enumerate(zip(cues,scheduled)):
        if not math.isfinite(duration) or duration<=0:raise ValueError('Generated duration must be positive')
        next_start=cues[index+1]['start'] if index+1<len(cues) else end
        limit=min(end,next_start-gap_seconds)
        earliest=max(origin,previous+(gap_seconds if result else 0),cue['start']-max_early_seconds)
        target=max(earliest,min(cue['start'],limit-duration))
        stop=target+duration
        overrun=stop>limit+.0001
        if overrun:issues.append({'kind':'speech_timing_overrun','cue':cue['id'],
            'seconds':duration,'available_seconds':max(0,limit-earliest)})
        result.append({'start':target,'end':stop,'source_start':cue['start'],
                       'source_end':cue['end'],'early_seconds':max(0,cue['start']-target),
                       'speed':1.0,'timing_overrun':overrun})
        previous=stop
    if fits is not None:annotate_fits(result,issues,cues,fits)
    return result,issues


def _checked_fit(fit,duration=None):
    """The receipt's (factor, before, after) after structural checks; ValueError for a receipt that cannot be trusted."""
    try:
        factor,before,after=float(fit['factor']),float(fit['before_seconds']),float(fit['after_seconds'])
    except (KeyError,TypeError,ValueError,OverflowError) as error:
        raise ValueError('A take-fit receipt needs factor, before_seconds and after_seconds') from error
    if not all(math.isfinite(x) for x in (factor,before,after)) or factor<1 or before<=0 or after<=0:
        raise ValueError('A take-fit receipt must carry finite positive durations and a factor >= 1')
    if after>before/factor*1.03+.1:
        # atempo frames its output (a few ms to ~20 ms of tail); anything beyond take_fit's own slack is not a fit.
        raise ValueError('A take-fit receipt\'s fitted duration is inconsistent with its factor')
    if duration is not None and abs(before-float(duration))>_FIT_MATCH_SECONDS:
        raise ValueError('A take-fit receipt does not describe this line\'s generated duration')
    return factor,before,after


def fitted_durations(durations,fits):
    """The durations the scheduler must see: each fitted line's receipt ``after_seconds``, the others unchanged.

    Use this to feed an existing scheduler wrapper (schedule_around_protection, ...) that does not take ``fits``,
    then call annotate_fits on its (placements, issues)."""
    if len(fits)!=len(durations):raise ValueError('Each line needs its fit receipt or None')
    return [duration if fit is None else _checked_fit(fit,duration)[2] for duration,fit in zip(durations,fits)]


def annotate_fits(placements,issues,cues,fits):
    """Record the fits on a schedule that was computed from ``fitted_durations`` (in place; returns both lists).

    Each fitted line's row gets ``speed`` = applied factor and a ``fit`` summary. Each ``speech_timing_overrun``
    issue of a fitted cue gets ``fit_factor``, ``unfitted_seconds`` and ``overrun_seconds`` (what still exceeds the
    available time after the fit). Rows and issues of unfitted lines are not touched, so a schedule with no fits is
    exactly the previous output."""
    if not len(placements)==len(cues)==len(fits):raise ValueError('Each line needs its fit receipt or None')
    by_cue={}
    for row,cue,fit in zip(placements,cues,fits):
        if fit is None:continue
        factor,before,after=_checked_fit(fit)
        row['speed']=factor
        row['fit']={'factor':factor,'before_seconds':before,'after_seconds':after,
                    'method':fit.get('method','atempo'),'applied':factor>1}
        by_cue[cue['id']]=(factor,before)
    for issue in issues:
        if issue.get('kind')!='speech_timing_overrun' or issue.get('cue') not in by_cue:continue
        factor,before=by_cue[issue['cue']]
        issue['fit_factor']=factor
        issue['unfitted_seconds']=before
        seconds,available=issue.get('seconds'),issue.get('available_seconds')
        if isinstance(seconds,(int,float)) and isinstance(available,(int,float)):
            issue['overrun_seconds']=max(0.,seconds-available)
    return placements,issues


def available_slot_seconds(cue,next_start,*,origin,end,previous_stop=None,max_early_seconds=.75,gap_seconds=.04):
    """Seconds schedule_lines will actually offer this line: its ``available_seconds``, with the same arithmetic.

    ``next_start`` is the next cue's start (None for the last cue: the window ``end``); ``previous_stop`` is where the
    previous placed take ends (None for the first line). Use it to size a fit to the real scheduler room (a line may
    start up to ``max_early_seconds`` before its cue) instead of the plain cue-start slot of take_fit.slot_seconds."""
    following=end if next_start is None else next_start
    limit=min(end,following-gap_seconds)
    previous=origin if previous_stop is None else previous_stop+gap_seconds
    earliest=max(origin,previous,cue['start']-max_early_seconds)
    return max(0,limit-earliest)
