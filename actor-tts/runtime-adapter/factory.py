"""Factory for the movie actor renderer: root-owned GPU admission (scope vox-actor-tts) and movie-Vox host options.
Configuration only; no model is started by this factory. The owned child uses the pinned CUDA backend.

Vox only (owner decision 2026-10-17: "I think switching to vox is best"). actor_beta.actor_engine may be absent or
"vox"; any other value (including the former "breeze") is a configuration error. No Breeze code is imported from this
package; A's frozen Breeze adapter stays on disk unchanged for rollback only. This package is imported as
`vox_actor_adapter` (automatic-v1 loads it), so its modules never collide with A's same-named top-level modules.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import approved
from .errors import ActorUnavailable, wrap_fault
from .vox_renderer import (DEFAULTS, VoxActorRenderer, concurrency_mode, configure_roots, max_instances,
                           vox_options)

HERE = Path(__file__).resolve().parent
SCOPE = 'vox-actor-tts'
ENGINES = ('vox',)
DEFAULT_ENGINE = 'vox'             # owner decision 2026-10-17: movie dubbing is Vox only

__all__ = ['RootAdmission', 'load_admission', 'make_renderer', 'actor_engine', 'host_options', 'SCOPE']


def actor_engine(config):
    """'vox' (actor_beta.actor_engine absent or "vox"). Anything else is a configuration error: Breeze was removed from
    the movie path; rolling back means reinstalling the previous relay files, not a flag."""
    beta = config.get('actor_beta') if hasattr(config, 'get') else None
    value = (beta or {}).get('actor_engine', DEFAULT_ENGINE) if isinstance(beta, dict) else DEFAULT_ENGINE
    if value not in ENGINES:
        raise ActorUnavailable('configuration_invalid', 'actor_beta.actor_engine must be "vox" (movie dubbing is Vox only), not %r'
                               % (value,), fault_code='vox_engine_configuration')
    return value


def _parse_utc(value):
    if not isinstance(value, str) or not value:
        raise ValueError('expires_utc must be an ISO-8601 UTC string')
    parsed = datetime.fromisoformat(value[:-1] + '+00:00' if value.endswith('Z') else value)
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError('expires_utc must be UTC')
    return parsed


@dataclass(frozen=True)
class RootAdmission:
    granted: bool
    scope: str
    issued_by: str
    expires_utc: str

    def valid(self, now=None):
        if self.granted is not True or self.scope != SCOPE or self.issued_by != 'root':
            return False
        try:
            expires = _parse_utc(self.expires_utc)
        except ValueError:
            return False
        return (now or datetime.now(timezone.utc)) < expires


def load_admission(path, *, now=None):
    """Read a root-issued admission JSON; raise ValueError for anything not a live vox-actor-tts grant."""
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as error:
        raise ValueError('admission unreadable: %s' % (error,)) from error
    if not isinstance(data, dict):
        raise ValueError('admission must be a JSON object')
    if data.get('granted') is not True:
        raise ValueError('admission not granted')
    if data.get('scope') != SCOPE:
        raise ValueError('admission scope must be %r' % SCOPE)
    if data.get('issued_by') != 'root':
        raise ValueError('admission must be issued_by root')
    _parse_utc(data.get('expires_utc'))
    admission = RootAdmission(True, SCOPE, 'root', data['expires_utc'])
    if not admission.valid(now):
        raise ValueError('admission expired')
    return admission


def _relay_root(config):
    vox = vox_options(config)
    if vox.get('relay_root'):
        return str(Path(vox['relay_root']).resolve())
    module = sys.modules.get('relay')
    if module is not None and getattr(module, '__file__', None):
        return str(Path(module.__file__).resolve().parents[1])
    raise ActorUnavailable('configuration_invalid',
                           'movie Vox host needs the relay root (import relay first or set actor_beta.vox.relay_root)',
                           fault_code='vox_relay_root_missing')


def host_options(config):
    """VoxEngineHost options from the relay config (plain values only; shared_host keys on them)."""
    try:
        return _host_options(config)
    except ActorUnavailable:
        raise
    except Exception as error:   # noqa: BLE001 - invalid paths/numeric settings are permanent configuration faults
        raise wrap_fault(error, reason='configuration_invalid', fault_code='vox_host_configuration') from error


def _host_options(config):
    cfg = {**DEFAULTS, **{k: v for k, v in dict(config or {}).items() if v is not None}}
    vox = vox_options(cfg)
    if not cfg.get('actor_runtime_manifest'):
        raise ActorUnavailable('configuration_invalid',
                               'actor_runtime_manifest (the Vox runtime manifest on this machine) is not configured',
                               fault_code='vox_manifest_missing')
    manifest = Path(cfg['actor_runtime_manifest']).resolve()
    configure_roots(cfg, vox)
    cache = cfg.get('cache_dir')
    log_dir = vox.get('log_dir') or (str(Path(cache) / 'vox-movie-engine') if cache else None)
    # The child's ResourceBudget gets NO Sem status files: it would open status.json in every PCM callback, and any
    # open handle makes Sem's own status write fail. Sem priority is the movie GPU lease (host.bind_lease). The child
    # keeps the relay's own hold/coordination/process checks (immediate) and a SUSTAINED GPU/RAM reserve check
    # (gpu_sustain_seconds; review P2-5) with an explicit reserve_ram_mib.
    budget = {'status_files': [], 'reserve_ram_mib': int(vox['reserve_ram_mib']),
              'coordination_file': cfg.get('resource_coordination_file') or None,
              'hold_file': cfg.get('resource_hold_file') or None,
              'process_names': sorted(cfg.get('resource_process_names') or []),
              'require_process_absence': bool(cfg.get('resource_require_process_absence', False))}
    # Machine locations from config: the Vox engine interpreter (actor_beta.vox.python, else the relay 'python').
    return {'manifest': str(manifest), 'relay_root': _relay_root(cfg),
            'python': vox.get('python') or cfg.get('python') or sys.executable,
            'budget': budget, 'capacity_seconds': float(cfg['actor_reference_seconds']),
            'reserve_mib': int(cfg['actor_reserve_gpu_mib']), 'required_mib': int(cfg['actor_required_gpu_mib']),
            'idle_grace_seconds': float(vox['idle_grace_seconds']),
            'log_dir': log_dir, 'poll_seconds': float(vox['poll_seconds']),
            'cancel_ack_seconds': float(vox['cancel_ack_seconds']), 'request_timeout': float(vox['timeout_seconds']),
            'instances': max_instances(vox), 'hard_cancel_ms': float(vox['hard_cancel_ms']),
            'reserve_ram_mib': int(vox['reserve_ram_mib']), 'gpu_sustain_seconds': float(vox['gpu_sustain_seconds']),
            'sibling_pins': sibling_pins(vox),
            # auto: a fresh process bootstraps with 1 instance (admitted_concurrency raises the live limit later)
            'initial_limit': 1 if concurrency_mode(vox)[0] == 'auto' else max_instances(vox),
            'refusal_hold_seconds': float(vox['refusal_hold_seconds']),
            'process_owner': 'lease'}          # production: engines only inside the movie GPU lease v2 Job


def sibling_pins(vox):
    """The child's sibling modules' pins from actor_beta.vox.sources_sha256 (the child execs exactly these bytes)."""
    pins = vox.get('sources_sha256') or {}
    return {name: pins.get('runtime-adapter/' + name) for name in ('approved.py', 'child_watchdog.py')
            if pins.get('runtime-adapter/' + name)}


def make_renderer(config, owner, *, admission, reference_gate=None, transport=None, provenance=None):
    """Return a VoxActorRenderer. The shared movie-Vox host is only attached for a valid vox-actor-tts RootAdmission;
    otherwise rendering fails with configuration_invalid. A callable reference_gate (B's admitted
    source/actor association) and a callable provenance check (program-cut reference) are mandatory; there is no
    hash-only path."""
    actor_engine(config)
    if not callable(reference_gate):
        raise ActorUnavailable('configuration_invalid', 'make_renderer requires a reference_gate',
                               fault_code='vox_reference_gate_missing')
    if not callable(provenance):
        raise ActorUnavailable('configuration_invalid', 'make_renderer requires a program provenance check',
                               fault_code='vox_provenance_check_missing')
    admitted = isinstance(admission, RootAdmission) and admission.valid()
    if not admitted:
        admission = None
    if transport is None and admitted:
        from .vox_engine_host import shared_host
        try:
            transport = shared_host(host_options(config))
        except ActorUnavailable:
            raise
        except Exception as error:   # noqa: BLE001 - factory failures are configuration, never resource contention
            raise wrap_fault(error, reason='configuration_invalid', fault_code='vox_host_configuration') from error
    return VoxActorRenderer(config, owner, transport=transport, admission=admission, reference_gate=reference_gate,
                            provenance=provenance)


def approved_identity():
    """The approved first-dub identity this package is pinned to (for receipts and MANIFEST)."""
    return {'engine': approved.ENGINE, 'model_sha256': approved.MODEL_SHA256, 'quantization': approved.QUANTIZATION,
            'runtime_manifest_sha256': approved.RUNTIME_MANIFEST_SHA256, 'actor_voice_sha256': approved.ACTOR_VOICE_SHA256,
            'settings': dict(approved.SETTINGS), 'seeds': list(approved.SEEDS), 'cloning_mode': approved.CLONING_MODE,
            'supported_cloning_modes': ['reference-only', 'reference+transcript']}
