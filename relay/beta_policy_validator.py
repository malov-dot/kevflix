"""Original-audio policy validation with the separate actor_beta original-fallback ledger.

The base validate_original_policy() is unchanged and always runs. With no beta ledger this is exactly the base
call. With a ledger (actor_beta only), each fallback cue must be an existing plan source cue inside the render
range, never synthesized, with only cue-scoped fallback kinds. A cue the base policy also keeps (a base
short-original kind on the same cue) stays a base keep, checked against its exact source interval, and is not
duplicated as a candidate. Those cues are then presented to
the base validator as render-classified candidates (their exact source intervals), so the base recomputation,
window, layout, source-cue and full render-range accounting checks still apply. The base policy's kept list is
never altered: a beta fallback is not reported as a short-original keep, and it never becomes synthesis output.
"""
from .actor_beta_hook import CUE_FALLBACK_KINDS
from .prepared_actor import validate_original_policy


def validate_policy_with_beta_fallback(policy, *, beta_original_fallback, beta_enabled, origin, end,
                                       synthesis_cues, all_cues, render_start=None, render_end=None, **base):
    fallback = beta_original_fallback
    if not fallback:
        return validate_original_policy(policy, origin=origin, end=end, synthesis_cues=synthesis_cues,
                                        all_cues=all_cues, render_start=render_start, render_end=render_end, **base)
    if not beta_enabled:
        raise ValueError('A beta original-fallback ledger requires actor_beta to be enabled')
    if not isinstance(fallback, dict) or any(
            not isinstance(ident, str) or not ident or not isinstance(kinds, list) or not kinds
            or any(kind not in CUE_FALLBACK_KINDS for kind in kinds) for ident, kinds in fallback.items()):
        raise ValueError('Beta original-fallback ledger is malformed')
    if not isinstance(policy, dict) or not isinstance(all_cues, list):
        raise ValueError('Beta original fallback needs the policy receipt and all source cues')
    start = origin if render_start is None else render_start
    stop = end if render_end is None else render_end
    source = {cue.get('id'): cue for cue in all_cues if isinstance(cue, dict)}
    synthesis = {cue['id'] for cue in synthesis_cues}
    kept = set(policy.get('preserved_cue_ids') or ())
    candidates = []
    for ident in sorted(fallback):
        cue = source.get(ident)
        if cue is None:
            raise ValueError('A beta original-fallback cue is not one of the plan source cues')
        if ident in synthesis:
            raise ValueError('A beta original-fallback cue cannot also be synthesized')
        if not (cue['end'] > start and cue['start'] < stop):
            raise ValueError('A beta original-fallback cue is outside the render range')
        if ident in kept:
            # Legitimate intersection: a beta-split kind plus a base short-original kind on one cue. The base keep
            # stands as recomputed below; its decision must describe this exact source cue interval.
            decision = next((d for d in policy.get('decisions') or () if isinstance(d, dict) and d.get('id') == ident), {})
            if ((decision.get('cue') or {}).get('start'), (decision.get('cue') or {}).get('end')) != (cue['start'], cue['end']):
                raise ValueError('A base-kept beta original-fallback cue does not match its source interval')
            continue
        candidates.append({'id': ident, 'start': cue['start'], 'end': cue['end']})
    return validate_original_policy(policy, origin=origin, end=end,
                                    synthesis_cues=list(synthesis_cues) + candidates, all_cues=all_cues,
                                    render_start=render_start, render_end=render_end, **base)
