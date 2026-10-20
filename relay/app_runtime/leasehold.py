"""Verified loader for the bundled movie GPU lease v2 client (CONTRACT section 2 / 2b).

relay/app_runtime/lease/lease_client.py is the byte-identical upstream file (author pin 5cbac8a4...; also covered
by the app integrity manifest). load_client() execs exactly the bytes whose sha256 equals LEASE_SHA256, once per
process, with __file__ set to the real path (the client resolves its tolerant reader and its watcher bootstrap from
it). Consumers: relay.app_runtime.movie_lease (the holder inside the server) and relay.app_runtime.lease_client.
"""
from __future__ import annotations

import hashlib
import sys
import threading
import types
from pathlib import Path

LEASE_PATH = Path(__file__).resolve().parent / 'lease' / 'lease_client.py'
LEASE_SHA256 = '5cbac8a4d33c114bf811bab50e5957142dfa46fa8e749ab482bf78749fefcd29'
_LOCK = threading.Lock()


class LeasePinRefused(Exception):
    pass


def require_pin(pin=None):
    pin = pin or {'path': LEASE_PATH, 'sha256': LEASE_SHA256}
    if not pin.get('sha256') or len(str(pin['sha256'])) != 64:
        raise LeasePinRefused('lease client has no sha256 pin')
    return pin


def load_client(pin=None):
    """Exec lease_client.py from the bytes whose sha256 equals the pin (read once: no TOCTOU, no __pycache__)."""
    pin = require_pin(pin)
    path = Path(pin['path'])
    with _LOCK:
        name = '_kevflix_lease_client_%s' % pin['sha256'][:12]
        if name in sys.modules:
            return sys.modules[name]
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != pin['sha256']:
            raise LeasePinRefused('lease_client.py sha256 %s != pin %s' % (digest, pin['sha256']))
        module = types.ModuleType(name)
        module.__file__ = str(path)
        sys.modules[name] = module                   # dataclasses resolve through sys.modules
        previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
        try:
            exec(compile(data, str(path), 'exec'), module.__dict__)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        finally:
            sys.dont_write_bytecode = previous
        return module
