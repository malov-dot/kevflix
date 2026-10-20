"""Release verified actor audio progressively without exposing future edits."""
from __future__ import annotations
import hashlib,json,math,time
from pathlib import Path
from .media import atomic_json,file_hash
from .actor_render import ActorRenderer
from .verified_speech import VerifiedSpeech
from .language import LanguageTools
from .cinematic import replacement_envelope,replace_dialogue,actor_mix_levels
from .prepared_actor import load_prepared_actor,validate_original_policy
from .original_audio_policy import protect_dialogue_mix,original_protection_masks
from .dubbing_timing import plan_dubbing_timing,place_timed_lines,check_timing_carry
from .dialogue_ledger import (merge_spans,persisted_protection_issues,protection_rows,DialogueLedger,
                              SampleCarryLedger,add_utterance,make_sample_carry,verify_sample_carry,
                              sample_carry_max_frames,PROTECTION_TRANSITION_SECONDS)
from .live_mux import LiveMux

LAYOUT_CHANNELS={'mono':1,'stereo':2,'5.1':6,'5.1(side)':6}


def live_mux_geometry(plan):
    """Channel count and layout a LiveMux must declare for a prepared plan.

    Taken from the plan's resolved ``channel_layout``; never from a count.
    """
    layout=plan.get('channel_layout')
    if layout not in LAYOUT_CHANNELS:
        raise ValueError('Prepared plan lacks an explicit supported channel layout; reprepare before live muxing')
    report=plan['report']
    return {'channels':LAYOUT_CHANNELS[layout],'channel_layout':layout,
            'origin':float(report['origin']),'end':float(report['end'])}


def open_live_mux(config,plan_path,source,folder,owner,*,output_channels=None):
    """LiveMux for a prepared plan, using ``plan['channel_layout']``."""
    import json
    plan=json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if plan.get('engine_fingerprint')!=config.engine_fingerprint():
        raise ValueError('Prepared actor plan is stale; prepare it with the current engine')
    geometry=live_mux_geometry(plan)
    return LiveMux(config['ffmpeg'],source,folder,geometry['origin'],geometry['end'],geometry['channels'],owner,
                   channel_layout=geometry['channel_layout'],output_channels=output_channels)


def _source_coverage(report,replacements,origin,end):
    """Prove that observed source speech has a known treatment before release.

    This checks supplied analysis evidence, not acoustic quality. Old prepared
    reports without explicit raw_turns are not silently interpreted as silence.
    Validated kept-original intervals count as known treatment; skipping a cue
    without such an explicit protected interval is insufficient.
    """
    regions=report.get('raw_turns')
    if not isinstance(regions,(list,tuple)):
        raise ValueError('Prepared source speech coverage is missing; reprepare with raw_turns evidence before streaming')
    occupied=sorted(replacements)
    uncovered=[]
    for region in regions:
        start,stop=region['start'],region['end']
        if isinstance(start,bool) or isinstance(stop,bool) or not all(
            isinstance(value,(int,float)) and math.isfinite(value) for value in (start,stop)
        ) or stop<=start:
            raise ValueError('Prepared source speech coverage has an invalid interval')
        left,right=max(origin,start),min(end,stop)
        if right<=left:continue
        cursor=left
        for begin,finish in occupied:
            if finish<=cursor or begin>=right:continue
            if begin>cursor+1e-9:uncovered.append((cursor,min(begin,right)))
            cursor=max(cursor,min(finish,right))
            if cursor>=right:break
        if cursor<right-1e-9:uncovered.append((cursor,right))
    if uncovered:
        first=uncovered[0]
        raise ValueError(f'Uncovered source speech at {first[0]:.6f}..{first[1]:.6f}; resolve its transcript or explicit original-audio preservation before streaming')
    return {'basis':'prepared-raw-speech-regions','observed_regions':len(regions),
            'uncovered_regions':0,'replacement_intervals':len(occupied)}


def _span_rows(rows,label):
    result=[]
    for row in rows:
        start,stop=row.get('start'),row.get('end')
        if isinstance(start,bool) or isinstance(stop,bool) or not all(
                isinstance(value,(int,float)) and math.isfinite(value) for value in (start,stop)) or stop<=start:
            raise ValueError(f'Next-window {label} has an invalid interval')
        result.append(dict(row))
    return result


def plan_digest(plan):
    """Canonical content digest of a parsed prepared plan (B6b binding).

    Taken over the parsed object, so formatting or key order never matters
    and the renderer can check exactly the object it loaded.
    """
    return hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()


def _sample_request(timing_carry,protection_ledger):
    """B5 opt-in: validate a sample-carry request before any media or PCM."""
    if set(timing_carry)-{'scope','previous','samples','lookahead','session','chain'}:
        raise ValueError('Sample carry accepts only scope, previous, samples, lookahead, session and chain')
    chain=timing_carry.get('chain',False)
    if not isinstance(chain,bool):
        raise ValueError('Sample carry chain must be true or false')
    if protection_ledger is None:
        raise ValueError('Sample carry needs the job protection ledger (A7 scope and canonical takes)')
    session=timing_carry['session']
    if not isinstance(session,str) or not session or len(session)>200:
        raise ValueError('Sample carry needs a nonempty output stream session')
    previous,incoming=timing_carry.get('previous'),timing_carry.get('samples')
    if previous is None and incoming is not None:
        raise ValueError('Sample carry needs its previous timing carry')
    if previous is not None and incoming is None:
        # Never assume a previous window released nothing past its edge.
        raise ValueError('The previous window\'s pending edge samples were not supplied (sample_carry_missing)')
    lookahead=timing_carry.get('lookahead')
    if lookahead is not None:
        if not isinstance(lookahead,dict) or set(lookahead)-{'end','speech','protected','pauses'}:
            raise ValueError('Next-window lookahead accepts only end, speech, protected and pauses')
        if 'speech' not in lookahead:
            raise ValueError('Next-window speech is unknown (sample_carry_next_speech_unknown)')
        if 'protected' not in lookahead:
            raise ValueError('Next-window original protection is unknown (sample_carry_next_protection_unknown)')
        end=lookahead.get('end')
        if isinstance(end,bool) or not isinstance(end,(int,float)) or not math.isfinite(end):
            raise ValueError('Next-window lookahead needs a finite end')
        lookahead={'end':float(end),'speech':_span_rows(lookahead['speech'],'speech'),
                   'protected':_span_rows(lookahead['protected'],'protection'),
                   'pauses':_span_rows(lookahead.get('pauses',()),'confirmed pause')}
    return {'session':session,'incoming':incoming,'lookahead':lookahead,'chain':chain}


def next_window_lookahead(next_plan,*,edge,channel_layout=None,protection_scope=None):
    """Lookahead for the window ending at ``edge`` from the adjacent prepared plan.

    Speech is the next plan's complete raw speech regions and source cues;
    protection is its kept-original policy spans and persisted keeps. A plan
    without raw speech evidence, or from another layout/scope, is refused.
    Confirmed global pauses are not inferred here; a caller may add them.
    """
    report=next_plan['report']
    if abs(float(report['origin'])-float(edge))>1e-6:
        raise ValueError('Next-window plan is not adjacent to this edge (sample_carry_lookahead_not_adjacent)')
    turns=report.get('raw_turns')
    if not isinstance(turns,list):
        raise ValueError('Next-window speech coverage is unknown (sample_carry_next_coverage_unknown)')
    if channel_layout is not None and next_plan.get('channel_layout')!=channel_layout:
        raise ValueError('Next-window plan has another channel layout')
    if protection_scope is not None and next_plan.get('protection_scope')!=protection_scope:
        raise ValueError('Next-window plan has another protection scope')
    speech=[{'start':t['start'],'end':t['end'],'speaker':t.get('speaker')} for t in turns]
    speech.extend({'start':c['start'],'end':c['end'],'speaker':c.get('speaker')}
                  for c in (next_plan.get('all_cues') or next_plan.get('cues') or []))
    protected=[{'cue':r['cue'],'start':r['start'],'end':r['end']}
               for r in (next_plan.get('original_audio_policy') or {}).get('protected_intervals',())]
    protected.extend({'cue':r.get('cue'),'start':r['start'],'end':r['end']}
                     for r in next_plan.get('persisted_original_protection') or [])
    return {'end':float(report['end']),'speech':speech,'protected':protected}


def _apply_incoming(incoming,timing,carries,ledger,session,first_frame,rate,layout,source,generated,intervals,protected):
    """Verify, check and claim the previous window's edge output before any PCM.

    Returns ``{'state': receipt}`` when invalidated (nothing is released), or
    the claim record whose carried frames are now mixed into ``generated``.
    """
    reason=None
    if timing['status']!='applied':
        reason=timing['reason'] or timing['status']
    elif incoming['session']!=session:
        reason='stream_session_changed'
    elif (incoming['job_id'],incoming['revision'],incoming['scope_key'])!=(ledger.job_id,ledger.revision,ledger.key):
        reason='protection_scope_changed'
    elif incoming['edge_frame']!=first_frame:
        reason='edge_frame_mismatch'
    elif incoming['rate']!=rate or incoming['channel_layout']!=layout or incoming['channels']!=source.shape[1]:
        reason='layout_changed'
    if reason is not None:
        # Seek, replay, gap, revision/source/scope or stream change: the
        # pending samples are stale and are never released anywhere.
        carries.discard(incoming)
        return {'state':{'status':'invalidated','reason':reason,'frames':incoming['frames'],'release':'discarded'}}
    removals=[(float(a),float(b)) for a,b in incoming['removal_intervals']]
    extent=max([incoming['frames']]+[round(b*rate)-first_frame for _,b in removals])
    if extent>len(source):
        raise ValueError('Carried edge output exceeds this window (sample_carry_exceeds_window)')
    pad=PROTECTION_TRANSITION_SECONDS
    for a,b in removals:
        if any(min(b+pad,q)>max(a-pad,p) for p,q in protected):
            raise ValueError('Carried dialogue removal would touch protected original audio '
                             '(carried_removal_crosses_protected_original)')
    tail=carries.tail(incoming)
    if protected and len(tail):
        _,allowed=original_protection_masks(len(tail),rate,first_frame/rate,protected)
        if (allowed<1).any():
            raise ValueError('Carried generated speech would touch protected original audio '
                             '(carried_tail_crosses_protected_original)')
    token,problems=carries.claim(incoming)
    if problems:
        raise ValueError('Carried edge samples cannot be released: '+', '.join(i['kind'] for i in problems))
    generated[:len(tail)]+=tail
    intervals.extend(removals)
    # ``token`` fences this claim (B6): only its holder may begin, confirm or
    # fail the release; a later claim on another connection supersedes it.
    claim={'carry':incoming,'extent':extent,'token':token,'begun':False,'settled':False,
           'state':{'status':'applied','reason':None,'frames':len(tail),'bytes':int(tail.nbytes),
                    'takes':[t['id'] for t in incoming['takes']],
                    'removal_intervals':[list(r) for r in removals],'extent_frames':extent,'release':'claimed'}}
    if extent==0:
        if not (carries.begin_release(incoming,token) and carries.confirm(incoming,token)):
            raise ValueError('Carried edge claim was superseded by another consumer (sample_carry_claim_superseded)')
        claim['begun']=claim['settled']=True;claim['state']['release']='released'
    return claim


def _commit_crossing_take(ledger,cue,line,placement,pan,gain):
    """Make a take that crosses the window edge canonical before release.

    Uses the A7 commit (generated take vs every persisted kept span, read in
    one IMMEDIATE transaction). An existing canonical take for the same cue
    with other samples or timing is a conflict, never overwritten.
    """
    digest=file_hash(line['audio'])
    if line.get('sha256',digest)!=digest:
        raise ValueError('Verified take file differs from its receipt')
    line['sha256']=digest
    ident,begin,stop,item=DialogueLedger(ledger.store,ledger.job_id,ledger.revision).entry(cue,line,placement,pan,gain)
    def conflicts(existing,takes,rows,utterances):
        problems=ledger.conflicts(existing,takes,rows,utterances)
        for take in takes:
            if (take.get('id')==ident and take.get('ledger_version')==1 and take.get('revision')==ledger.revision
                    and (take.get('sha256')!=item['sha256'] or take.get('start')!=item['start'])):
                problems.append({'kind':'sample_carry_canonical_take_conflict','cue':ident})
        return problems
    problems=ledger.store.commit_protection(ledger.job_id,ledger.revision,ledger.key,[],[(ident,begin,stop,item)],
                                            conflicts=conflicts)
    if problems:
        raise ValueError('A take crossing the window edge conflicts with persisted state: '+
                         ', '.join(sorted({i['kind'] for i in problems})))
    return item


def render_progressively(config,plan_path,source_wav,owner,sink,*,language=None,hints=(),event=lambda _:None,
                         protection_ledger=None,timing_carry=None,bound_plan=None):
    """``protection_ledger`` (a dialogue_ledger.ProtectionLedger for this job,
    revision and scope) makes persisted kept-original decisions binding and
    commits this window's own keeps before the first PCM block is released.
    Without it only the plan's own (and prepare-time persisted) spans apply.

    ``timing_carry`` (opt-in) is ``{'scope': timing_carry_scope(...),
    'previous': prior receipt['timing_carry']['carry_out'] or None}``. Pass the
    previous carry only after that adjacent window fully succeeded. Lookahead
    is refused unless the B5 sample carry is requested.

    B5 sample carry (opt-in; docs/MOVIE-SAMPLE-CARRY-B5.md) adds ``session``
    (the continuous output stream), ``samples`` (prior receipt
    ``['sample_carry']``; required whenever ``previous`` is given) and
    ``lookahead`` (``next_window_lookahead(next_plan, edge=end)``). It needs
    ``protection_ledger``. A take may then cross ``end`` into the observed
    lookahead; its samples past ``end`` are held back, committed as a
    canonical take, and released exactly once by the adjacent next window.

    B6b (controller only): ``bound_plan`` is the ``plan_digest`` of the plan
    the caller validated. The object actually loaded must match it before
    any ledger commit, file write or PCM; a later reread can never change
    the rendered window. ``timing_carry['chain']=True`` turns an invalidated
    incoming carry into a failure before PCM instead of a silent discard.
    """
    import numpy as np
    import soundfile as sf
    owner.check()
    sample=None
    if timing_carry is not None:
        # Checked before media, ledger commits or any release.
        if not isinstance(timing_carry,dict):raise ValueError('Timing carry must be a mapping')
        if 'session' in timing_carry:
            sample=_sample_request(timing_carry,protection_ledger)
        else:
            if 'lookahead' in timing_carry:
                raise ValueError('Timing lookahead needs cross-window sample carry; pass a stream session to enable it')
            if set(timing_carry)-{'scope','previous'}:
                raise ValueError('Timing carry accepts only scope and previous here')
        carry_scope=check_timing_carry(timing_carry.get('scope'),timing_carry.get('previous'))
        if protection_ledger is not None and (carry_scope['revision']!=protection_ledger.revision or any(
                carry_scope.get(k)!=protection_ledger.scope.get(k) for k in
                ('source_fingerprint','audio_index','channel_layout','engine_fingerprint'))):
            raise ValueError('Timing carry scope differs from the protection ledger scope')
        if sample is not None and sample['incoming'] is not None:
            verify_sample_carry(sample['incoming'],timing_carry.get('previous'))
    plan_path=Path(plan_path)
    if bound_plan is None:
        plan,source,speech,rate=load_prepared_actor(config,plan_path,source_wav)
    else:
        try:
            plan,source,speech,rate=load_prepared_actor(config,plan_path,source_wav)
        except (OSError,RuntimeError,ValueError) as error:
            # Validated at the JSON level only; its inputs failed at load.
            raise ValueError(f'{error} (progressive_window_rejected_at_load)') from error
        if plan_digest(plan)!=bound_plan:
            raise ValueError('Prepared plan changed after the controller validated it '
                             '(progressive_plan_changed_since_validation)')
    if timing_carry is not None and carry_scope['channel_layout']!=plan.get('channel_layout'):
        raise ValueError('Timing carry layout differs from the prepared plan layout')
    report=plan['report'];origin,end=report['origin'],report['end']
    if sample is not None:
        carries=SampleCarryLedger(protection_ledger,sample['session'])
        first_frame=round(origin*rate);edge_frame=first_frame+len(source)
        # One stream never continues past the same edge twice (replay/seek
        # needs a new session); refused before any commit or PCM.
        carries.check_open(edge_frame)
    cues=plan['cues'];voices=plan['voices']
    # Raw diarization labels are not accepted actor IDs. Treat their speech as
    # unknown competing audio when deciding whether a translated line has room.
    replacements={cue['id']:(cue['start']-.12,cue['end']+.25) for cue in cues}
    # Every protected original span is known before the first block. Plans
    # without a policy keep the previous unprotected mix exactly.
    policy=plan.get('original_audio_policy')
    layout=plan.get('channel_layout')
    # Eligibility is recomputed from the plan's source cues and raw speech
    # regions; a saved keep decision is never trusted on its own.
    protected=validate_original_policy(policy,origin=origin,end=end,synthesis_cues=cues,
        channel_layout=layout,channels=source.shape[1],all_cues=plan.get('all_cues'),
        speech_regions=report.get('raw_turns')) if 'original_audio_policy' in plan else []
    # Coverage counts only this plan's own validated decisions; persisted
    # spans from other windows constrain output but never prove coverage.
    coverage=_source_coverage(report,list(replacements.values())+protected,origin,end)
    own_protected=protected
    persisted=list(plan.get('persisted_original_protection') or [])
    if protection_ledger is not None:
        if plan.get('protection_scope')!=protection_ledger.scope:
            raise ValueError('Prepared plan protection scope differs from this job; reprepare before streaming')
        fresh=protection_ledger.between(origin,end,window=(origin,end),original=source,rate=rate)
        if fresh['issues']:
            raise ValueError('Persisted original protection blocks streaming: '+
                             ', '.join(sorted({i['kind'] for i in fresh['issues']})))
        persisted=fresh['active']
    if persisted:
        kept=set((policy or {}).get('preserved_cue_ids',()))
        selected=[c for c in plan.get('all_cues') or cues if c['id'] in kept]+list(cues)
        crossing=persisted_protection_issues(persisted,selected,policy or {'protected_intervals':[]})
        if crossing:
            raise ValueError('A synthesis or kept cue crosses persisted original protection; resolve before streaming')
        protected=merge_spans(protected,[(r['start'],r['end']) for r in persisted])
    if protection_ledger is not None and policy:
        # Decisions are known to every other window before this one's first block.
        problems=protection_ledger.commit(protection_rows(policy,source,origin,rate,window_end=end,
            render_start=origin,render_end=end,layout=layout))
        if problems:
            raise ValueError('Kept-original decision conflicts with persisted state: '+
                             ', '.join(sorted({i['kind'] for i in problems})))
    lookahead=None if sample is None else sample['lookahead']
    ahead_protected=[];walls=protected
    if lookahead is not None:
        # Next-window keeps are walls for tails and removal pads crossing the
        # edge: persisted rows (re-read now) plus the next plan's own keeps.
        if lookahead['end']>end:
            ahead=protection_ledger.between(end,lookahead['end'])
            if ahead['issues']:
                raise ValueError('Next-window original protection is undecided: '+
                                 ', '.join(sorted({i['kind'] for i in ahead['issues']})))
            ahead_protected=[{'cue':r['cue'],'start':r['start'],'end':r['end']} for r in ahead['active']]
        ahead_protected+=[{'cue':r.get('cue'),'start':r['start'],'end':r['end']} for r in lookahead['protected']]
        if persisted_protection_issues(ahead_protected,list(cues)):
            raise ValueError('A synthesis cue or its removal crosses next-window original protection '
                             '(sample_carry_removal_crosses_protected_original)')
        spans=[(r['start'],r['end']) for r in ahead_protected]
        # Spans whose transition reaches into this window change its mix.
        protected=merge_spans(protected,[s for s in spans if s[0]<end+PROTECTION_TRANSITION_SECONDS])
        walls=merge_spans(protected,spans)
    regions=[{'start':t['start'],'end':t['end']} for t in report['raw_turns']]
    regions.extend({'start':a,'end':b} for a,b in protected)
    carry=None
    if timing_carry is not None:
        # Persisted keeps (ledger or prepare-time) are walls with their transition.
        carry={'scope':carry_scope,'previous':timing_carry.get('previous'),
               'protected':[{'cue':r.get('cue'),'start':r['start'],'end':r['end']} for r in persisted]}
        if lookahead is not None:
            carry['protected']+=ahead_protected
            carry['lookahead']={'end':lookahead['end'],'speech':lookahead['speech']}
    timing_plan=plan_dubbing_timing(cues,origin=origin,end=end,speech_regions=regions,carry=carry,
                                    pause_windows=() if lookahead is None else lookahead['pauses'])
    atomic_json(plan_path.parent/'progressive-timing-plan.json',timing_plan)
    if timing_plan['issues']:
        raise ValueError('Resolve source timing and translated pause alignment before streaming')
    slots={cue['id']:cue for cue in timing_plan['cues']}
    renderer=None
    generated=np.zeros_like(source);intervals=[];lines=[];emitted=0;unconfirmed_sink_write=None
    started=time.perf_counter();block_frames=2*rate;metrics=[]
    incoming_state=claim=None;outgoing=[];outgoing_removals=[]
    if sample is not None:
        incoming_state={'status':'none'}
        incoming=sample['incoming']
        if incoming is not None:
            claim=_apply_incoming(incoming,timing_plan['carry'],carries,protection_ledger,sample['session'],
                                  first_frame,rate,layout,source,generated,intervals,protected)
            incoming_state=claim['state']
            if incoming_state['status']!='applied':
                claim=None
                if sample['chain']:
                    # A controller chain is one continuous stream: a broken
                    # edge stops it here, before this window's first block.
                    raise ValueError(f'Chained incoming carry was invalidated ({incoming_state["reason"]}); '
                                     'the stream stops at this edge (progressive_chain_carry_invalidated)')
    def begin_carried_write(start_frame):
        # At most once (B6 fencing): the first block holding a carried frame
        # is written only after this claim's token moves claimed->releasing.
        if claim is not None and not claim['begun'] and start_frame<claim['extent']:
            if not carries.begin_release(claim['carry'],claim['token']):
                raise ValueError('Carried edge claim was superseded by another consumer '
                                 '(sample_carry_claim_superseded)')
            claim['begun']=True;incoming_state['release']='releasing'
    def settle_released():
        if claim is not None and claim['begun'] and not claim['settled']:
            if not carries.confirm(claim['carry'],claim['token']):
                raise ValueError('Carried edge release lost its fencing token (sample_carry_claim_superseded)')
            claim['settled']=True;incoming_state['release']='released'
    def release(stop,final=False):
        nonlocal emitted,unconfirmed_sink_write
        right=len(source) if final and stop==end else min(len(source),round(stop*rate)-round(origin*rate))
        if not final:right=(right//block_frames)*block_frames
        while emitted<right:
            owner.check()
            limit=min(right,emitted+block_frames)
            at=origin+emitted/rate
            envelope=replacement_envelope(limit-emitted,rate,at,intervals)
            if protected:
                mixed,measurement=protect_dialogue_mix(source[emitted:limit],speech[emitted:limit],generated[emitted:limit],
                    envelope,origin=(round(origin*rate)+emitted)/rate,rate=rate,channel_layout=layout,protected_intervals=protected)
                if measurement['requires_rescheduling']:
                    raise ValueError('Generated speech touches protected original audio; reschedule before release')
            else:
                mixed,measurement=replace_dialogue(source[emitted:limit],speech[emitted:limit],generated[emitted:limit],envelope)
            if measurement['overrange_samples']:raise ValueError('Live audio exceeds its output range')
            # Cancellation during mixing must not reach the sink.
            owner.check()
            # A sink may partially write before raising. Record that unknown
            # interval so a caller never treats a failed pipe as safe to resume.
            if claim is not None:begin_carried_write(emitted)
            unconfirmed_sink_write={'start_frame':emitted,'end_frame':limit}
            sink(mixed)
            if claim is not None:settle_released()
            emitted=limit;unconfirmed_sink_write=None;metrics.append(measurement)
            event({'kind':'pcm_released','elapsed':time.perf_counter()-started,'through':origin+emitted/rate})
    try:
        # A proven-safe intro can fill the player buffer while the first actor
        # take is prepared. A window with no observed speech needs no model.
        frontier=min((min(slots[c['id']]['earliest_start'],replacements[c['id']][0]) for c in cues),default=end)
        release(frontier,final=not cues)
        if cues:
            owner.check()
            renderer=ActorRenderer(config,owner)
            renderer.prepare_references(voices.values())
            verifier=VerifiedSpeech(renderer,language or LanguageTools(config,'cpu'),owner,hints)
        for index,cue in enumerate(cues):
            owner.check()
            event({'kind':'voice_started','cue':cue['id'],'elapsed':time.perf_counter()-started})
            slot=slots[cue['id']]
            line=verifier.render(cue,voices[cue['speaker']],plan_path.parent/'live-utterances',slot['budget_seconds'])
            owner.check()
            if line.get('cue')!=cue['id']:
                raise ValueError('Verified actor take changed its canonical cue identity')
            placements,issues=place_timed_lines({'cues':[slot],'issues':slot['issues']},{cue['id']:line['seconds']})
            if issues and sample is not None and placements[0]['end']>end:
                # Explicit reason when the take needs the next window's evidence.
                reason=('sample_carry_lookahead_missing' if lookahead is None else
                        'sample_carry_lookahead_insufficient' if placements[0]['end']>lookahead['end'] else
                        'sample_carry_tail_blocked_by_next_speech_or_protection')
                raise ValueError(f'Verified speech cannot fit without changing released audio ({reason})')
            if issues:raise ValueError('Verified speech cannot fit without changing released audio')
            placement=placements[0]
            wave,voice_rate=sf.read(line['audio'],dtype='float32')
            if voice_rate!=rate or wave.ndim!=1 or not len(wave) or not np.isfinite(wave).all():
                raise ValueError('Actor output rate or samples changed')
            if abs(len(wave)/rate-line['seconds'])>1/rate:
                raise ValueError('Actor output duration differs from its receipt')
            left=round(placement['start']*rate)-round(origin*rate)
            crossing=left+len(wave)-len(source)
            if left<emitted or (crossing>0 and sample is None):raise ValueError('Actor output crosses the committed timeline')
            if left+len(wave)>round(slot['latest_end']*rate)-round(origin*rate):
                raise ValueError('Actor samples cross their source timing budget')
            if crossing>sample_carry_max_frames(rate):
                raise ValueError('Generated speech past the window edge exceeds the bounded sample carry')
            if walls:
                _,allowed=original_protection_masks(len(wave),rate,(round(origin*rate)+left)/rate,walls)
                if (allowed<1).any():
                    raise ValueError('Verified speech would touch protected original audio')
            if max(0,round(replacements[cue['id']][0]*rate)-round(origin*rate))<emitted:
                raise ValueError('Dialogue removal would modify already committed audio')
            pan,gain,levels=actor_mix_levels(speech,wave,cue,origin,rate,lfe_index=3 if source.shape[1]==6 else None)
            line['mix_levels']=levels
            if crossing>0:
                # The take becomes canonical before any of it is released, so
                # the next window rebuilds the exact same samples past ``end``.
                item=_commit_crossing_take(protection_ledger,cue,line,placement,pan,gain)
                add_utterance(generated,item,origin,rate)
                outgoing.append(item)
            else:
                generated[left:left+len(wave)]+=wave[:,None]*pan[None,:]*gain
            intervals.append(replacements[cue['id']])
            if sample is not None and round(replacements[cue['id']][1]*rate)>edge_frame:
                outgoing_removals.append(replacements[cue['id']])
            line['placement']=placement;lines.append(line)
            event({'kind':'voice_verified','cue':cue['id'],'elapsed':time.perf_counter()-started,'cache_hit':line['cache_hit']})
            # Future takes now start at their source timestamps. Their removal
            # envelopes begin .12s earlier; neither region may be committed yet.
            pending=cues[index+1:]
            frontier=min((min(slots[c['id']]['earliest_start'],replacements[c['id']][0]) for c in pending),default=end)
            release(frontier,final=not pending)
        release(end,final=True)
        if sample is not None:
            if claim is not None and not claim['settled']:
                raise ValueError('Carried edge samples were not released')
            # Published only after every frame of this window reached the sink.
            outgoing_carry=make_sample_carry(protection_ledger,session=sample['session'],
                timing_carry_out=timing_plan['carry']['carry_out'],edge=end,edge_frame=edge_frame,rate=rate,
                layout=layout,items=outgoing,removal_intervals=outgoing_removals)
            problems=carries.publish(outgoing_carry)
            if problems:
                raise ValueError('Edge carry conflicts with this stream: '+', '.join(i['kind'] for i in problems))
        result={'lines':lines,'frames':emitted,'seconds':time.perf_counter()-started,'mix_blocks':metrics,
                'all_lines_readback_passed':True,'timing_policy':timing_plan['version'],
                'source_coverage':coverage,
                'channel_layout':layout,
                'original_audio_protection':None if not protected else {
                    'version':(policy or {}).get('version'),'preserved_cue_ids':(policy or {}).get('preserved_cue_ids',[]),
                    'protected_intervals':[list(i) for i in own_protected],
                    'applied_protected_intervals':[list(i) for i in protected],
                    'covered_issues':(policy or {}).get('covered_issues',[]),
                    'cross_window_persistence':protection_ledger is not None,
                    'human_listening_acceptance':False},
                'quality_scope':'Readback and timeline checks only; residual dialogue, soundtrack fidelity and sustained throughput remain unverified',
                'original_outside_replacements_preserved':all(m['original_preserved_outside_replacements'] for m in metrics)}
        if timing_carry is not None:
            # Hand carry_out to the next adjacent window only after this success.
            result['timing_carry']=timing_plan['carry']
        if sample is not None:
            result['sample_carry']=outgoing_carry
            result['sample_carry_in']=incoming_state
        atomic_json(plan_path.parent/'progressive-actor-receipt.json',result)
        return result
    except BaseException as error:
        failure={'error':str(error),'committed_frames':emitted,'verified_cues':[line['cue'] for line in lines],
            'unconfirmed_sink_write':unconfirmed_sink_write,
            'resume_same_sink_allowed':False,
            'timing_policy':timing_plan['version'],'seconds':time.perf_counter()-started}
        if sample is not None:
            if claim is not None and claim['begun'] and not claim['settled']:
                # A partial sink write may have carried frames: never replay
                # them. Only the token holder can record this (B6).
                if carries.unconfirm(claim['carry'],claim['token']):
                    incoming_state['release']='unconfirmed'
                claim['settled']=True
            # Owned unreleased output (this window's edge carry) is discarded
            # by never publishing it; released or canonical state is kept.
            failure['sample_carry']={'incoming':incoming_state,'outgoing_published':False}
        atomic_json(plan_path.parent/'progressive-actor-failure.json',failure)
        raise
    finally:
        if renderer is not None:renderer.close()
