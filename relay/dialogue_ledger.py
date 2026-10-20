"""Stable actor utterances across chunk boundaries and seeks."""
from __future__ import annotations
import hashlib
import json
import math
import uuid
from pathlib import Path
from .media import file_hash
from .language import words


class DialogueLedger:
    def __init__(self,store,job_id,revision=1):
        self.store,self.job_id,self.revision=store,job_id,revision

    def between(self,start,end):
        result=[]
        for row in self.store.utterances(self.job_id,start,end):
            item=row['payload']
            if item.get('ledger_version')!=1 or item.get('revision')!=self.revision:continue
            path=Path(item['audio'])
            if not path.is_file() or file_hash(path)!=item['sha256']:
                raise ValueError('A canonical actor utterance changed')
            result.append(item)
        return result

    def match(self,cue,items):
        matches=[]
        for item in items:
            if item['speaker']!=cue['speaker'] or words(item['text'])!=words(cue['english']):continue
            shared=max(0,min(item['source_end'],cue['end'])-max(item['source_start'],cue['start']))
            shorter=min(item['source_end']-item['source_start'],cue['end']-cue['start'])
            if shorter>0 and shared/shorter>=.7:matches.append((shared,item))
        if len(matches)>1:raise ValueError('Multiple canonical actor lines match this cue')
        return matches[0][1] if matches else None

    def entry(self,cue,line,placement,pan,gain):
        """The (id, begin, end, payload) row put() stores, without storing it."""
        item={**line,**placement,'ledger_version':1,'revision':self.revision,
              'id':cue['id'],'cue':cue['id'],'speaker':cue['speaker'],'text':cue['english'],
              'source_text':cue['text'],'source_start':cue['start'],'source_end':cue['end'],
              'channel_weights':list(map(float,pan)),'voice_gain':float(gain),
              'replacement_start':cue['start']-REPLACEMENT_BEFORE_SECONDS,
              'replacement_end':cue['end']+REPLACEMENT_AFTER_SECONDS}
        begin=min(item['start'],item['replacement_start'])
        end=max(item['end'],item['replacement_end'])
        return item['id'],begin,end,item

    def put(self,cue,line,placement,pan,gain):
        ident,begin,end,item=self.entry(cue,line,placement,pan,gain)
        self.store.put_utterance(self.job_id,ident,begin,end,item)
        return item


def take_span(item):
    """Complete source, placement and removal extent of a canonical take."""
    return (min(item['start'],item['replacement_start'],item['source_start']),
            max(item['end'],item['replacement_end'],item['source_end']))


PROTECTION_LEDGER_VERSION='protected-original-ledger-v1'
# Same 10 ms edit transition protect_dialogue_mix places outside each span.
PROTECTION_TRANSITION_SECONDS=.01
# Removal padding DialogueLedger.put records around each generated cue.
REPLACEMENT_BEFORE_SECONDS,REPLACEMENT_AFTER_SECONDS=.12,.25


def protection_scope(*,source_fingerprint,audio_index,channel_layout,engine_fingerprint,policy_version):
    """Everything a kept-original decision depends on besides job and revision."""
    scope={'source_fingerprint':source_fingerprint,'audio_index':audio_index,
           'channel_layout':channel_layout,'engine_fingerprint':engine_fingerprint,
           'policy_version':policy_version,'ledger_version':PROTECTION_LEDGER_VERSION}
    if not isinstance(source_fingerprint,str) or not source_fingerprint:
        raise ValueError('Protected-original scope needs the source fingerprint')
    if isinstance(audio_index,bool) or not isinstance(audio_index,int):
        raise ValueError('Protected-original scope needs the selected audio stream index')
    if channel_layout not in ('mono','stereo','5.1','5.1(side)'):
        raise ValueError('Protected-original scope needs an explicit channel layout')
    if not engine_fingerprint or not policy_version:
        raise ValueError('Protected-original scope needs the engine fingerprint and policy version')
    return scope


def scope_key(scope):
    return hashlib.sha256(json.dumps(scope,sort_keys=True).encode()).hexdigest()


def span_pcm_sha256(original,origin,rate,start,end):
    """Hash of the exact source cells protected for [start,end), or None if outside."""
    import numpy as np
    first=round(origin*rate)
    left=math.floor(start*rate+1e-9)-first;right=math.ceil(end*rate-1e-9)-first
    if left<0 or right>len(original) or right<=left:return None
    return hashlib.sha256(np.ascontiguousarray(original[left:right]).tobytes()).hexdigest()


def same_span(a,b,tolerance=1e-6):
    """Identical absolute span up to float noise (far below one 48 kHz sample)."""
    return a is not None and b is not None and abs(a[0]-b[0])<=tolerance and abs(a[1]-b[1])<=tolerance


def _touches(a,b,pad=PROTECTION_TRANSITION_SECONDS):
    return min(a[1]+pad,b[1])>max(a[0]-pad,b[0])


def _scope_issue(stored,scope,item):
    changed=sorted(k for k in set(stored)|set(scope) if stored.get(k)!=scope.get(k)) or ['scope_key']
    kind='protected_original_layout_incompatible' if changed==['channel_layout'] else 'protected_original_source_changed'
    return {'kind':kind,'protected_cue':item['cue'],'start':item['start'],'end':item['end'],
            'changed_scope_fields':changed}


def merge_spans(*groups):
    """Union of absolute (start,end) spans, sorted and coalesced."""
    spans=sorted((float(a),float(b)) for group in groups for a,b in group)
    merged=[]
    for begin,stop in spans:
        if merged and begin<=merged[-1][1]:merged[-1]=(merged[-1][0],max(merged[-1][1],stop))
        else:merged.append((begin,stop))
    return merged


def persisted_protection_issues(persisted,cues,policy=None):
    """Cue-scoped blockers for selected cues touching persisted kept spans.

    Without ``policy`` (before classification) only the identical cue of a
    persisted keep is exempt. With the final ``policy`` that cue must also be
    kept again with the identical interval: re-entry never dubs or subtracts
    a span another pass already published as original.
    """
    kept=None if policy is None else {r['cue']:(r['start'],r['end']) for r in policy.get('protected_intervals',())}
    issues=[]
    for row in persisted:
        span=(row['start'],row['end'])
        for cue in cues:
            # The cue's removal envelope (DialogueLedger.put pads) is subtraction too.
            extent=(cue['start']-REPLACEMENT_BEFORE_SECONDS,cue['end']+REPLACEMENT_AFTER_SECONDS)
            if not _touches(span,extent):continue
            # The kept interval may be wider than the cue (parent/source bounds);
            # identity is the cue ID, and the final policy must keep that span.
            if cue['id']==row['cue'] and (kept is None or same_span(kept.get(cue['id']),span)):
                continue
            issue={'kind':'cue_crosses_protected_original','cue':cue['id'],'protected_cue':row['cue'],
                   'start':row['start'],'end':row['end'],'cue_start':cue['start'],'cue_end':cue['end']}
            if issue not in issues:issues.append(issue)
    return issues


def protection_rows(policy,original,origin,rate,*,window_end,render_start,render_end,layout):
    """Ledger rows for this window's own kept-original decisions."""
    decisions={d['id']:d for d in policy.get('decisions',())}
    covered={}
    for issue in policy.get('covered_issues',()):covered.setdefault(issue['cue'],[]).append(issue)
    rows=[]
    for row in policy.get('protected_intervals',()):
        decision=decisions[row['cue']]
        rows.append({'cue':row['cue'],'start':row['start'],'end':row['end'],'basis':row.get('basis'),
                     'policy_version':policy['version'],'natural_policy':policy['natural_policy'],
                     'source_text':(decision.get('cue') or {}).get('text'),
                     'covered_issues':covered.get(row['cue'],[]),'channel_layout':layout,
                     'window':{'origin':float(origin),'end':float(window_end),
                               'render_start':float(render_start),'render_end':float(render_end)},
                     'span_pcm_sha256':span_pcm_sha256(original,origin,rate,row['start'],row['end']),
                     'human_listening_acceptance':False})
    return rows


class ProtectionLedger:
    """Kept-original decisions persisted across windows, seeks and restarts.

    Separate from DialogueLedger: rows hold no generated PCM or English. A row
    is binding only for the same job, revision and scope (source fingerprint,
    audio stream, channel layout, engine fingerprint and policy version).
    Older-revision rows are invalidated (ignored). A newer revision, another
    source/engine/policy scope or another layout in the same revision is an
    explicit unscoped blocker, never a silent drop.
    """
    def __init__(self,store,job_id,revision,scope):
        self.store,self.job_id,self.revision=store,job_id,int(revision)
        self.scope=dict(scope);self.key=scope_key(self.scope)

    def between(self,start,end,*,window=None,original=None,rate=None):
        """Rows touching [start,end). ``window`` (origin,end) and decoded
        ``original`` PCM let a re-entered window prove its source is unchanged."""
        active,stale,issues,evidence=[],[],[],[]
        for row in self.store.protected_originals(self.job_id,start-1.,end+1.):
            if not _touches((row['start'],row['end']),(start,end)):continue
            item={**row['payload'],'cue':row['cue_id'],'start':row['start'],'end':row['end'],'revision':row['revision']}
            if row['revision']<self.revision:
                stale.append(item);continue
            if row['revision']>self.revision:
                issues.append({'kind':'protected_original_revision_changed','protected_cue':item['cue'],
                               'start':item['start'],'end':item['end'],'stored_revision':row['revision'],
                               'revision':self.revision})
                continue
            if row['scope_key']!=self.key:
                issues.append(_scope_issue(item.get('scope') or {},self.scope,item));continue
            if original is not None and window is not None:
                current=span_pcm_sha256(original,window[0],rate,item['start'],item['end'])
                same_window=[item.get('window',{}).get(k) for k in ('origin','end')]==[float(window[0]),float(window[1])]
                record={'protected_cue':item['cue'],'same_window':same_window,
                        'pcm_compared':current is not None,'pcm_matches':current==item.get('span_pcm_sha256')}
                evidence.append(record)
                if same_window and current!=item.get('span_pcm_sha256'):
                    # A re-entered window decodes the same samples or its source changed.
                    issues.append({'kind':'protected_original_source_changed','protected_cue':item['cue'],
                                   'start':item['start'],'end':item['end'],'changed_scope_fields':['span_pcm_sha256']})
                    continue
            active.append(item)
        return {'version':PROTECTION_LEDGER_VERSION,'active':active,'stale_revision':stale,
                'issues':issues,'pcm_evidence':evidence}

    def conflicts(self,existing,takes,rows,utterances):
        problems=[]
        for row in existing:
            if row['revision']>self.revision:
                problems.append({'kind':'protected_original_revision_changed','protected_cue':row['cue_id'],
                                 'stored_revision':row['revision'],'revision':self.revision})
        current=[r for r in existing if r['revision']==self.revision]
        new_takes=[u[3] for u in utterances]
        for row in rows:
            span=(row['start'],row['end'])
            for other in current:
                if not _touches(span,(other['start'],other['end'])) and other['cue_id']!=row['cue']:continue
                if other['scope_key']!=self.key:
                    stored={**other['payload'],'cue':other['cue_id'],'start':other['start'],'end':other['end']}
                    problems.append({**_scope_issue(stored.get('scope') or {},self.scope,stored),'cue':row['cue']})
                elif other['cue_id']!=row['cue'] or not same_span((other['start'],other['end']),span):
                    problems.append({'kind':'protected_original_decision_conflict','cue':row['cue'],
                                     'protected_cue':other['cue_id'],'start':row['start'],'end':row['end'],
                                     'stored_start':other['start'],'stored_end':other['end']})
            for take in takes+new_takes:
                if take.get('ledger_version')!=1 or take.get('revision')!=self.revision:continue
                if _touches(span,take_span(take)):
                    problems.append({'kind':'kept_original_conflicts_with_canonical_take','cue':row['cue'],
                                     'canonical_id':take['id'],'start':row['start'],'end':row['end']})
        protected=[(r['start'],r['end'],r['cue_id']) for r in current]+[(r['start'],r['end'],r['cue']) for r in rows]
        for take in new_takes:
            for begin,stop,cue in protected:
                if _touches((begin,stop),take_span(take)):
                    problems.append({'kind':'generated_take_conflicts_with_protected_original','cue':take['id'],
                                     'protected_cue':cue,'start':begin,'end':stop})
        unique=[]
        for problem in problems:
            if problem not in unique:unique.append(problem)
        return unique

    def commit(self,rows,utterances=()):
        """Persist kept spans (and this window's new takes) atomically.

        Returns conflict issues; nothing is stored when any exist. Must run
        before the window's output is written or released.
        """
        stored=[{'cue':r['cue'],'start':float(r['start']),'end':float(r['end']),
                 'payload':{**r,'scope':self.scope,'revision':self.revision}} for r in rows]
        return self.store.commit_protection(self.job_id,self.revision,self.key,stored,list(utterances),
                                            conflicts=self.conflicts)


def add_utterance(generated,item,origin,rate):
    """Add only the visible samples; preserve exact PCM/timing across windows."""
    import numpy as np
    import soundfile as sf
    samples,sample_rate=sf.read(item['audio'],dtype='float32')
    if sample_rate!=rate or samples.ndim!=1:raise ValueError('Canonical actor audio geometry changed')
    pan=np.asarray(item['channel_weights'],dtype=np.float32)
    if len(pan)!=generated.shape[1]:raise ValueError('Canonical actor channel layout changed')
    offset=round(item['start']*rate)-round(origin*rate)
    left=max(0,offset);source_left=max(0,-offset)
    count=min(len(generated)-left,len(samples)-source_left)
    if count>0:
        generated[left:left+count]+=samples[source_left:source_left+count,None]*pan[None,:]*item['voice_gain']
    return max(0,count)


# ---------------------------------------------------------------------------
# B5 sample carry (opt-in): generated PCM that crosses a progressive window
# edge. Contract: docs/MOVIE-SAMPLE-CARRY-B5.md. The samples themselves are
# never stored here; a carry references canonical DialogueLedger takes and the
# SHA-256 of their exact tail, rebuilt with add_utterance on both sides.
SAMPLE_CARRY_VERSION='sample-carry-v1'
# Bound on carried generated PCM. The timing policy allows at most 1 s of tail.
SAMPLE_CARRY_MAX_SECONDS=1.
SAMPLE_CARRY_MAX_TAKES=8
SAMPLE_CARRY_MAX_REMOVALS=32
SAMPLE_CARRY_STATES=('pending','claimed','releasing','released','unconfirmed','discarded')
_SAMPLE_CARRY_KEYS=frozenset({'version','job_id','revision','scope_key','session','timing_digest',
                              'edge','edge_frame','rate','channel_layout','channels','frames','bytes',
                              'takes','removal_intervals','pcm_sha256','digest'})
_LAYOUT_CHANNELS={'mono':1,'stereo':2,'5.1':6,'5.1(side)':6}


def _carry_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def sample_carry_max_frames(rate):
    return int(math.ceil(SAMPLE_CARRY_MAX_SECONDS*rate))


def _finite(value,label):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError(f'Sample carry {label} must be a finite number')
    return float(value)


def _count(value,label,low=0,high=None):
    if isinstance(value,bool) or not isinstance(value,int) or value<low or (high is not None and value>high):
        raise ValueError(f'Sample carry {label} is out of range')
    return value


def take_tail(items,edge_frame,rate,channels):
    """Exact generated PCM at and after absolute ``edge_frame``.

    Built with add_utterance, the same arithmetic the producing window used
    for the samples before the edge, so the two halves are one take.
    """
    import numpy as np
    import soundfile as sf
    origin=edge_frame/rate
    if round(origin*rate)!=edge_frame:raise ValueError('Sample carry edge is not representable')
    frames=max([round(item['start']*rate)+sf.info(item['audio']).frames-edge_frame for item in items]+[0])
    if frames>sample_carry_max_frames(rate):
        raise ValueError('Generated speech past the window edge exceeds the bounded sample carry')
    tail=np.zeros((frames,channels),dtype=np.float32)
    for item in items:add_utterance(tail,item,origin,rate)
    return tail


def make_sample_carry(ledger,*,session,timing_carry_out,edge,edge_frame,rate,layout,items,removal_intervals):
    """Metadata for this window's pending edge output (no PCM inside).

    ``ledger`` is the job's ProtectionLedger (A7 scope key, job, revision).
    ``items`` are canonical takes already committed through it.
    """
    channels=_LAYOUT_CHANNELS[layout]
    tail=take_tail(items,edge_frame,rate,channels)
    carry={'version':SAMPLE_CARRY_VERSION,'job_id':ledger.job_id,'revision':ledger.revision,
           'scope_key':ledger.key,'session':session,
           'timing_digest':None if timing_carry_out is None else timing_carry_out['digest'],
           'edge':float(edge),'edge_frame':int(edge_frame),'rate':int(rate),'channel_layout':layout,
           'channels':channels,'frames':len(tail),'bytes':int(tail.nbytes),
           'takes':[{'id':i['id'],'sha256':i['sha256'],'start':float(i['start']),
                     'start_frame':round(i['start']*rate)} for i in items],
           'removal_intervals':[[float(a),float(b)] for a,b in removal_intervals],
           'pcm_sha256':hashlib.sha256(tail.tobytes()).hexdigest()}
    carry['digest']=_carry_digest(carry)
    return verify_sample_carry(carry)


def verify_sample_carry(carry,previous=None):
    """A restored carry must be exactly what a window emitted; never repaired."""
    if not isinstance(carry,dict) or set(carry)!=_SAMPLE_CARRY_KEYS:
        raise ValueError('Sample carry is malformed')
    if carry['version']!=SAMPLE_CARRY_VERSION:
        raise ValueError('Sample carry has an unsupported version')
    try:
        expected=_carry_digest({k:v for k,v in carry.items() if k!='digest'})
    except (TypeError,ValueError) as error:
        raise ValueError('Sample carry is not finite JSON data') from error
    if carry['digest']!=expected:
        raise ValueError('Sample carry digest does not match its content')
    for field in ('job_id','scope_key','session','pcm_sha256'):
        if not isinstance(carry[field],str) or not carry[field]:
            raise ValueError(f'Sample carry lacks its {field}')
    _count(carry['revision'],'revision',1)
    rate=_count(carry['rate'],'rate',1,384000)
    _count(carry['edge_frame'],'edge frame')
    _finite(carry['edge'],'edge')
    if _LAYOUT_CHANNELS.get(carry['channel_layout'])!=carry['channels']:
        raise ValueError('Sample carry needs an explicit channel layout matching its channels')
    frames=_count(carry['frames'],'frame count',0,sample_carry_max_frames(rate))
    if carry['bytes']!=frames*carry['channels']*4:
        raise ValueError('Sample carry byte count does not match its frames')
    if not isinstance(carry['takes'],list) or len(carry['takes'])>SAMPLE_CARRY_MAX_TAKES:
        raise ValueError('Sample carry take list is out of range')
    if bool(frames)!=bool(carry['takes']):
        raise ValueError('Sample carry frames and takes disagree')
    for take in carry['takes']:
        if not isinstance(take,dict) or set(take)!={'id','sha256','start','start_frame'}:
            raise ValueError('Sample carry take reference is malformed')
        if not isinstance(take['id'],str) or not take['id'] or not isinstance(take['sha256'],str):
            raise ValueError('Sample carry take reference lacks its identity')
        _finite(take['start'],'take start')
        if _count(take['start_frame'],'take start frame')>=carry['edge_frame']:
            raise ValueError('A carried take must start before its edge')
    if not isinstance(carry['removal_intervals'],list) or len(carry['removal_intervals'])>SAMPLE_CARRY_MAX_REMOVALS:
        raise ValueError('Sample carry removal list is out of range')
    for row in carry['removal_intervals']:
        if not isinstance(row,list) or len(row)!=2 or _finite(row[1],'removal end')<=_finite(row[0],'removal start'):
            raise ValueError('Sample carry removal interval is invalid')
    if previous is not None and carry['timing_digest']!=previous.get('digest'):
        raise ValueError('Sample carry does not belong to the previous timing carry')
    return carry


class SampleCarryLedger:
    """At-most-once release markers for carried edge PCM, per output stream.

    Keyed by job, revision, A7 scope key, stream ``session`` and absolute
    edge frame. States: pending (published by a successful window), claimed
    (a consumer will mix it), releasing (the token holder is writing the
    first block that contains a carried frame), released (that write
    returned), unconfirmed (it failed; frames may have reached the sink) and
    discarded (invalidated by a seek, revision, scope or session change).

    B6 fencing: ``claim`` returns a fencing token that is strictly larger
    than any earlier token for the same row. ``begin_release``, ``confirm``
    and ``unconfirm`` require the current token and the expected state in one
    conditional SQLite UPDATE, so of two consumers on separate connections at
    most one ever writes a carried frame. A later claim of a ``claimed`` row
    (retry or takeover) fences the earlier claimant before it wrote anything;
    ``releasing``, ``released``, ``unconfirmed`` and ``discarded`` rows can
    never be claimed again. Fencing lives in the database, not in this object.
    """
    def __init__(self,protection_ledger,session,claimant=None):
        if not isinstance(session,str) or not session or len(session)>200:
            raise ValueError('Sample carry needs a nonempty output stream session')
        self.session=session
        self.claimant=claimant or uuid.uuid4().hex
        if not isinstance(self.claimant,str) or len(self.claimant)>200:
            raise ValueError('Sample carry claimant must be a short string')
        self.store,self.job_id,self.revision,self.key=(protection_ledger.store,protection_ledger.job_id,
                                                       protection_ledger.revision,protection_ledger.key)

    @staticmethod
    def _key(carry):
        return (carry['job_id'],carry['revision'],carry['scope_key'],carry['session'],carry['edge_frame'])

    def state(self,edge_frame):
        row=self.store.sample_carry(self.job_id,self.revision,self.key,self.session,edge_frame)
        return None if row is None else row['state']

    def check_open(self,edge_frame):
        """This stream has not yet continued past ``edge_frame``."""
        state=self.state(edge_frame)
        if state is not None:
            raise ValueError(f'This output stream already produced its edge carry ({state}); '
                             'start a new stream session after a seek or replay')

    def publish(self,carry):
        state=self.store.publish_sample_carry(*self._key(carry),carry['digest'],carry)
        return [] if state is None else [{'kind':'sample_carry_conflict','state':state}]

    def claim(self,carry):
        """``(token, problems)``; ``token`` is None whenever ``problems``."""
        result=self.store.claim_sample_carry(*self._key(carry),carry['digest'],self.claimant)
        if result['token'] is not None:
            return result['token'],[]
        state=result['state']
        if state is None:
            return None,[{'kind':'sample_carry_unpublished'}]
        if not result['digest_matches']:
            return None,[{'kind':'sample_carry_conflict','state':state}]
        return None,[{'kind':{'releasing':'sample_carry_release_in_progress',
                              'released':'sample_carry_already_released',
                              'unconfirmed':'sample_carry_release_unconfirmed',
                              'discarded':'sample_carry_discarded'}.get(state,'sample_carry_conflict'),
                      'state':state}]

    def _advance(self,carry,token,from_states,to_state):
        return self.store.advance_sample_carry(*self._key(carry),carry['digest'],token,from_states,to_state)

    def begin_release(self,carry,token):
        """Fenced step immediately before the first sink write with a carried frame."""
        return self._advance(carry,token,('claimed',),'releasing')

    def confirm(self,carry,token):
        """The sink write returned: the carried frames are released."""
        return self._advance(carry,token,('releasing',),'released')

    def unconfirm(self,carry,token):
        """The sink write failed after ``begin_release``: never replay it."""
        return self._advance(carry,token,('releasing',),'unconfirmed')

    def discard(self,carry):
        """Invalidate a pending carry; claimed or settled rows are never rewritten."""
        return self.store.discard_sample_carry(*self._key(carry),carry['digest'])

    def tail(self,carry):
        """Rebuild and verify the carried PCM from its canonical takes."""
        from .media import file_hash
        items=[]
        for ref in carry['takes']:
            rows=[r for r in self.store.utterances(carry['job_id'],ref['start']-1e-6,ref['start']+1e-6)
                  if r['id']==ref['id']]
            item=rows[0]['payload'] if rows else None
            if (item is None or item.get('ledger_version')!=1 or item.get('revision')!=carry['revision']
                    or item.get('sha256')!=ref['sha256'] or item.get('start')!=ref['start']
                    or not Path(item['audio']).is_file() or file_hash(item['audio'])!=item['sha256']):
                raise ValueError('A carried canonical take changed or is missing (sample_carry_take_changed)')
            items.append(item)
        tail=take_tail(items,carry['edge_frame'],carry['rate'],carry['channels'])
        if len(tail)!=carry['frames'] or hashlib.sha256(tail.tobytes()).hexdigest()!=carry['pcm_sha256']:
            raise ValueError('Carried generated PCM differs from its receipt (sample_carry_pcm_changed)')
        return tail
