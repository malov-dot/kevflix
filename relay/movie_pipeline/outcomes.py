"""Window outcome classification and the original-audio window publication (relay.movie_pipeline, W10).

Why this exists. Under the serial coordinator.loop rule, one window that raised (settled 'error') or stopped at
needs_review makes Coordinator.summarize mark the WHOLE job error/needs_review. Store.claim then refuses every pending
chunk of the job, and TVService/audio_chunk block playback. The approved behaviour for NORMAL dialogue uncertainty is
original audio for that window, and production continues. Original speech is never removed unless a completed English
replacement covers it.

classify(error) -> (kind, reason). Recoverable is an ALLOWLIST; everything else is explicit.
  'cancelled'    Cancelled family, including beta_line_ledger.Superseded: settled 'cancelled' (coordinator.loop rule).
  'resource'     ResourceYield / InterruptedError / ActorBusy: settled 'waiting_resources'; retry_resources requeues it.
  'recoverable'  only these named outcomes:
                   ActorUnavailable with a cue reason; DeadlineExpired; LineClaimLost / InaudibleTakes /
                   LineCommitRace;
                   exact relay ValueError messages for unschedulable or untranslatable lines;
                   a pipeline-decided needs_review (intercepted by TrialClaimedStore).
                 These go through publish_original.
  'fatal'        everything else, including unknown and programming errors: settled 'error' with the reason.

publish_original (fix 5). The window plays its ORIGINAL audio, except where the central line ledger already holds
COMMITTED English takes. These are canonical DialogueLedger takes, typically the tail of a line owned and committed by
the previous window and carried across the boundary; they keep playing:

  * Carried takes are mixed exactly as ActorPipeline mixes a carry:
    add_utterance on the window's absolute timeline (origin), replacement_envelope (10 ms fades) over the speech-only
    removal intervals of each take (carry_removal_intervals: the take's unpadded source span analysed in the removal
    stem, merged), and protect_dialogue_mix over the first-pass processing audio with its removal stem. Original speech
    is removed ONLY while the carried take's source is speaking; pauses and all other speech, music and effects stay
    original.
  * The carried takes are already published by their owner window. This window only plays their part inside its own
    render range, so nothing is mixed twice. A v4 window record (mode 'original-fallback', mixed_canonical_ids) is
    committed under the chunk fence, so the ledger knows what this window rendered.
  * A READY window never drops an intersecting committed English take. When the carry cannot be mixed exactly, the
    window is NOT published: no verified first-pass bundle, or the ledger kept changing, raises CarryPending (a
    ResourceYield, so the window is settled waiting_resources and retried); a geometry or mix refusal, or ledger
    protection problems, raises CarryRefused (an explicit error). The window record and the receipt name the ACTUAL
    mixed utterance ids. The record is committed only without protection problems or races.
  * english.wav, the carry mix and receipt.json are written under the v4 chunk fence (current claim token +
    revision), from per-claim temporary names. The chunk is then settled 'ready' through coordinator._ClaimedStore
    (token-checked). The mode is 'original-trial-fallback', with zero NEW dubbed credit.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

VERSION = 'kevflix-original-window-v2'
CUE_REASONS = frozenset({'reference_unverified', 'reference_invalid', 'untrusted_actor', 'text_invalid', 'output_invalid'})
RECOVERABLE_TYPES = frozenset({'DeadlineExpired', 'LineClaimLost', 'InaudibleTakes', 'LineCommitRace'})
RECOVERABLE_MESSAGES = ('A synthesis cue overlaps protected original audio', 'A synthesis cue was not scheduled',
                        'No English dialogue was produced', 'Resolve the speaker before synthesizing this line')
LEDGER_ATTEMPTS = 2
CARRY_FADE_SECONDS = .01      # the 10 ms fades ActorPipeline uses for speech-only removal intervals

from ..resource_budget import ResourceYield as _ResourceYield   # noqa: E402  (stdlib-only module)


class CarryPending(_ResourceYield):
    """A committed English take crosses this window but its removal data or a stable ledger snapshot is not available
    now: explicit waiting_resources (Store.retry_resources requeues the window), never a ready window without it."""


class CarryRefused(RuntimeError):
    """A committed English take crosses this window and cannot be mixed exactly (geometry, mix refusal, ledger
    protection problems): explicit error state; never hidden as an original-audio success."""


def classify(error):
    from ..movie_faults import fault_record
    from ..process import Cancelled
    from ..resource_budget import ResourceYield
    fault = fault_record(error)
    if fault is not None:
        return fault['kind'], '%s: %s' % (fault['code'], fault['detail'][:800])
    name = type(error).__name__
    reason = '%s: %s' % (name, str(error)[:800])
    if isinstance(error, Cancelled) or name == 'Superseded':
        return 'cancelled', reason
    if isinstance(error, (ResourceYield, InterruptedError)) or name == 'ActorBusy':
        return 'resource', reason
    if name == 'ActorUnavailable' and getattr(error, 'reason', None) in CUE_REASONS:
        return 'recoverable', reason
    if name in RECOVERABLE_TYPES:
        return 'recoverable', reason
    if type(error) is ValueError and str(error).startswith(RECOVERABLE_MESSAGES):
        return 'recoverable', reason
    return 'fatal', reason


class NeedsReviewIntercepted(Exception):
    """The pipeline tried to settle this window needs_review; the coordinator publishes it original instead."""

    def __init__(self, result):
        super().__init__('window settled needs_review: %s' % sorted({
            str(i.get('kind')) for i in (result.get('issues') or []) if isinstance(i, dict)}))
        self.result = result


def trial_claimed_store(claimed_store_class):
    """coordinator._ClaimedStore whose own-chunk 'needs_review' settle is intercepted (nothing is written then)."""

    class TrialClaimedStore(claimed_store_class):
        def update_chunk(self, job_id, index, state, **updates):
            # The legacy actor wrapper can label a needs-review window ready
            # after substituting the raw original. In this pipeline it must
            # pass through publish_original so previously committed English
            # crossing this boundary is retained and the window is fenced.
            raw_original_fallback = (state in ('ready', 'approved')
                                     and updates.get('mode') == 'original-unverified'
                                     and updates.get('pipeline_state') == 'needs_review')
            if (job_id, int(index)) == self._key and (state == 'needs_review' or raw_original_fallback):
                raise NeedsReviewIntercepted(dict(updates))
            return super().update_chunk(job_id, index, state, **updates)

    return TrialClaimedStore


def _first_pass_arrays(directory, frames):
    """(bundle, arrays) of this window's verified first-pass bundle, or (None, reason)."""
    from .. import beta_first_pass_flow as _fpf
    path = Path(directory) / 'first-pass' / 'first-pass-bundle.json'
    if not path.is_file():
        return None, 'no first-pass bundle for this window'
    try:
        bundle = json.loads(path.read_text(encoding='utf-8'))
        return bundle, _fpf.arrays(bundle, frames)
    except (OSError, ValueError) as error:
        return None, 'first-pass bundle unusable: %s' % error


def carry_removal_intervals(removal, carry, origin, rate, lfe_index=None):
    """Absolute removal intervals for the carried canonical takes: speech-only, merged, ready for the 10 ms envelope.

    Pure and shared with the owner window: every take goes through cinematic.take_removal_intervals (its authorized
    source spans, else its recorded source span, else its recorded replacement interval), which analyses the UNPADDED
    source speech of the take in ``removal`` and falls back to the padded span only where the stem shows nothing. The same
    take on the same stem therefore gives the same intervals here as in ActorPipeline's owner window. Raises ValueError
    for a take that records no span."""
    from ..cinematic import merge_intervals, take_removal_intervals
    intervals = []
    for item in carry:
        intervals.extend(take_removal_intervals(removal, item, origin, rate, lfe_index))
    return merge_intervals(intervals)


def _carry_mix(store, claim, w, fence, tmp_name):
    """Mix the committed canonical takes over this window's render range. Returns (mixed_path or None, record)."""
    import soundfile as sf
    from .. import beta_dialogue_ownership as _own, beta_line_ledger as _lines
    from ..actor_pipeline import PROTECTION_TRANSITION_SECONDS, write_layout_wav
    from ..cinematic import replacement_envelope
    from ..dialogue_ledger import DialogueLedger, add_utterance
    from ..original_audio_policy import protect_dialogue_mix
    import numpy as np
    origin, end = float(w['origin']), float(w['end'])
    render_start, render_end = float(w['chunk']['start']), float(w['chunk']['end'])
    info = sf.info(str(w['original']))
    frames = int(info.frames)
    end = origin + frames / float(info.samplerate)
    record = {'version': VERSION, 'carried_ids': [], 'heard': False}
    for attempt in range(1, LEDGER_ATTEMPTS + 1):
        canonical = DialogueLedger(store, claim.job_id, claim.revision).between(origin, end)
        carry = [item for item in canonical if max(item['end'], item['replacement_end']) > render_start
                 and min(item['start'], item['replacement_start']) < render_end]
        record.update(carried_ids=[item['id'] for item in carry], attempt=attempt)
        key = _lines.window_key(render_start, render_end)
        window = {'window_key': key, 'render_start': render_start, 'render_end': render_end, 'origin': origin,
                  'end': end, 'new_take_end_limit': end, 'mixed_canonical_ids': [], 'new_take_ids': [],
                  'attempt': attempt, 'owned_unit_spans': [], 'mode': 'original-fallback'}
        mixed_path = None
        if carry:
            # Never publish a READY window that drops an intersecting COMMITTED English take: every branch that cannot
            # mix the carry exactly raises (CarryPending -> waiting_resources, CarryRefused -> error); no plain original.
            bundle, arrays = _first_pass_arrays(w['directory'], frames)
            if bundle is None:
                raise CarryPending('committed English take(s) %s cross this window but %s; the window waits for its '
                                   'removal data' % (record['carried_ids'], arrays))
            processing, removal = arrays['processing'], arrays['removal_stem']
            generated = np.zeros_like(processing)
            try:
                for item in carry:
                    add_utterance(generated, item, origin, info.samplerate)
                intervals = carry_removal_intervals(removal, carry, origin, info.samplerate)
            except ValueError as error:
                raise CarryRefused('committed English take(s) %s cannot be placed in this window: %s'
                                   % (record['carried_ids'], error)) from error
            envelope = replacement_envelope(len(processing), info.samplerate, origin, intervals,
                                            fade_seconds=CARRY_FADE_SECONDS)
            mixed, mix = protect_dialogue_mix(processing, removal, generated, envelope, origin=origin,
                                              rate=info.samplerate, channel_layout='stereo', protected_intervals=[],
                                              transition_seconds=PROTECTION_TRANSITION_SECONDS)
            if mix.get('requires_rescheduling') or mix.get('overrange_samples'):
                raise CarryRefused('carry mix of %s refused (rescheduling %s, overrange %s)' % (
                    record['carried_ids'], bool(mix.get('requires_rescheduling')),
                    int(mix.get('overrange_samples') or 0)))
            mixed_path = Path(w['directory']) / tmp_name
            _lines.fenced_write(store, fence, lambda: write_layout_wav(mixed_path, mixed, info.samplerate, 'stereo'))
            window['mixed_canonical_ids'] = [item['id'] for item in carry]      # the ACTUAL mixed utterance ids
            record.update(heard=True, mixed_ids=list(window['mixed_canonical_ids']),
                          mix={'requires_rescheduling': False, 'overrange_samples': 0})
        problems, races = _lines.commit(store, job_id=claim.job_id, revision=claim.revision, window=window,
                                        snapshot_ids=[item['id'] for item in canonical], utterances=[], line_claims=[],
                                        protected=[], same_line=_own.significant_overlap,
                                        claimant=_lines.new_claimant(key), fence=fence)
        record.update(window_record={'committed': not (problems or races), 'problems': problems, 'races': races,
                                     'mixed_canonical_ids': list(window['mixed_canonical_ids'])})
        if problems:
            if mixed_path is not None:
                mixed_path.unlink(missing_ok=True)
            raise CarryRefused('window record refused by the line ledger (protection problems): %s' % (problems,))
        if not races:
            return mixed_path, record
        if mixed_path is not None:
            mixed_path.unlink(missing_ok=True)          # a take was committed meanwhile: rebuild from the new snapshot
    raise CarryPending('the line ledger kept changing across %d attempts; the window waits for a stable snapshot'
                       % LEDGER_ATTEMPTS)


def publish_original(config, store, claim, owner, *, chunk_inputs, stage, reason, detail=None):
    """Settle THIS claimed window 'ready': original audio plus already committed English carried takes (fix 5)."""
    from .. import beta_line_ledger as ledger, coordinator as _coordinator
    from ..media import RATE, CHANNELS, atomic_json, extract_audio, file_hash
    fence = ledger.fence_of(claim.job_id, claim.index, claim.token, claim.revision)
    w = chunk_inputs(config, store, claim.job_id, claim.index)           # source/engine/subtitle checks
    ledger.check_fence(store, fence)
    directory, chunk, origin, spec = w['directory'], w['chunk'], w['origin'], w['spec']
    original = w['original']
    if not Path(original).is_file():
        extract_audio(config, owner, spec['source'], spec['audio_index'], origin, w['end'] - origin, original,
                      channels=w['channels'], preserve_layout=w['layout'])
    requested_end = chunk['end']
    decoded_end = None
    from ..worker import final_chunk
    if final_chunk(spec, claim.index):
        import soundfile as sf
        info = sf.info(str(original))
        if info.samplerate <= 0:
            raise CarryRefused('Final original audio has no valid sample rate')
        decoded_end = origin + info.frames / float(info.samplerate)
        if chunk['start'] < decoded_end < chunk['end']:
            # Match the actor-render path: the final audio can end before the
            # video/container. Carry records, output frames and stored timeline
            # must all use the same actual decoded end. The TV mux owns padding
            # the final video tail; this path does not invent source speech.
            chunk = {**chunk, 'end': decoded_end}
            w = {**w, 'chunk': chunk}
    previous = Path(directory) / 'receipt.json'
    kept = Path(directory) / ('receipt.blocked-t%d.json' % claim.token)
    if previous.is_file() and not kept.exists():
        ledger.fenced_write(store, fence, lambda: os.replace(previous, kept))
    mixed, carry = _carry_mix(store, claim, w, fence, 'original-carry.t%d.wav' % claim.token)
    source = mixed if mixed is not None else original
    ready = Path(directory) / 'english.wav'
    temporary = Path(directory) / ('english.t%d.fallback.tmp.wav' % claim.token)
    owner.run([config['ffmpeg'], '-hide_banner', '-loglevel', 'error', '-nostdin', '-ss', str(chunk['start'] - origin),
               '-i', source, '-t', str(chunk['end'] - chunk['start']), '-ar', RATE, '-ac', CHANNELS, '-c:a',
               'pcm_s16le', '-y', temporary])
    owner.check()
    ledger.fenced_write(store, fence, lambda: os.replace(temporary, ready))
    result = {'audio': str(ready), 'sha256': file_hash(ready), 'mode': 'original-trial-fallback',
              'pipeline_state': 'original_fallback', 'revision': claim.revision, 'chunk': claim.index,
              'start': chunk['start'], 'end': chunk['end'], 'source_fingerprint': spec['source_fingerprint'],
              'issues': [{'kind': 'original_window', 'stage': stage, 'reason': str(reason)[:1000]}],
              'original_window': {'version': VERSION, 'stage': stage, 'reason': str(reason)[:2000], 'detail': detail,
                                  'requested_render_end': requested_end, 'decoded_audio_end': decoded_end,
                                  'new_dubbed_credit_ms': 0, 'carried_english': carry,
                                  'audio': ('original audio with committed carried English takes'
                                            if mixed is not None else 'original source audio, unchanged'),
                                  'blocked_receipt': str(kept) if kept.exists() else None, 'at': time.time()},
              'quality': {'machine_checks_passed': False, 'human_listening_acceptance': False,
                          'scope': 'Original audio kept for this window (no completed new English replacement)'}}
    ledger.fenced_write(store, fence, lambda: atomic_json(Path(directory) / 'receipt.json', result))
    claimed = _coordinator._ClaimedStore(store, claim.job_id, claim.index, claim.token)
    claimed.update_chunk(claim.job_id, claim.index, 'ready', **result)
    store.event(claim.job_id, 'chunk', chunk=claim.index, state='ready', mode='original-trial-fallback', stage=stage)
    return result
