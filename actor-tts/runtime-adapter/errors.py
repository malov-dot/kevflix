"""Structured actor-TTS errors (INTERFACES.md section 2)."""
from __future__ import annotations

import sys
from pathlib import Path

REASONS = frozenset({
    'gpu_admission_hold', 'server_unavailable', 'model_unverified', 'reference_unverified',
    'reference_invalid', 'untrusted_actor', 'text_invalid', 'output_invalid',
    'configuration_invalid', 'runtime_failed',
})
FAULT_KINDS = frozenset({'fatal', 'resource', 'recoverable', 'cancelled'})
_FATAL_REASONS = frozenset({'server_unavailable', 'model_unverified', 'configuration_invalid', 'runtime_failed'})


def _resource_yield():
    try:
        from relay.resource_budget import ResourceYield
        return ResourceYield
    except Exception:
        pass
    root = Path(__file__).resolve().parents[3]
    if (root / 'relay' / 'resource_budget.py').is_file():
        if str(root) not in sys.path:
            sys.path.append(str(root))
        try:
            from relay.resource_budget import ResourceYield
            return ResourceYield
        except Exception:
            pass
    return RuntimeError


_Base = _resource_yield()


class ActorUnavailable(Exception):
    """Structured failure; only ``fault_kind == 'recoverable'`` permits cue fallback.

    ``reason`` stays compatible with the actor adapter contract. Schedulers must inspect fault_kind before legacy
    reason lists: unavailable runtime/configuration is fatal, never an indefinite GPU-resource wait.
    """

    def __init__(self, reason, detail=None, *, fault_kind=None, fault_code=None):
        if reason not in REASONS:
            raise ValueError('unknown ActorUnavailable reason: %r' % (reason,))
        default = 'fatal' if reason in _FATAL_REASONS else ('resource' if reason == 'gpu_admission_hold' else 'recoverable')
        self.fault_kind = fault_kind or default
        if self.fault_kind not in FAULT_KINDS:
            raise ValueError('unknown actor fault kind: %r' % (self.fault_kind,))
        self.fault_code = fault_code or 'vox_' + reason
        self.reason, self.detail = reason, detail
        super().__init__(reason if detail is None else '%s: %s' % (reason, detail))


def wrap_fault(error, *, reason='runtime_failed', fault_code=None, detail=None):
    """Preserve structured failures across adapter boundaries; unclassified failures are fatal."""
    kind = getattr(error, 'fault_kind', 'fatal')
    if kind not in FAULT_KINDS:
        kind = 'fatal'
    if kind == 'resource':
        reason = 'gpu_admission_hold'
    elif kind == 'recoverable' and reason in _FATAL_REASONS:
        reason = 'output_invalid'
    return ActorUnavailable(reason, str(error) if detail is None else detail, fault_kind=kind,
                            fault_code=getattr(error, 'fault_code', None) or fault_code or 'vox_' + reason)


class ActorCancelled(InterruptedError):
    """Caller cancellation, distinct from an unavailable GPU resource."""
    fault_kind = 'cancelled'

    def __init__(self, detail='movie Vox request cancelled', *, fault_code='vox_cancelled'):
        self.detail, self.fault_code = str(detail), fault_code
        super().__init__(self.detail)


class ActorBusy(_Base):
    """A real lease/VRAM hold; caller may retry after the resource becomes available."""

    def __init__(self, detail='movie Vox inference slot busy', *, fault_code='vox_resource_busy'):
        self.detail = detail
        self.fault_kind, self.fault_code = 'resource', fault_code
        super().__init__(detail)
