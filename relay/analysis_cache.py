"""Content-addressed storage for immutable, JSON-safe model observations.

This module deliberately imports no model, tensor, or array libraries.  Speaker
bindings and actor/reference-bank revisions belong to the caller's later decision
stage, not to keys for raw speech, diarization, or embedding observations.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import stat
import tempfile
import threading
import time
from typing import Any, Callable, Iterator


_FORMAT = "kevflix-analysis-cache-v1"
_MAX_ENTRY_BYTES = 64 * 1024 * 1024
_LOCK_WAIT_SECONDS = 0.2
_LOCK_COUNT = 256
_KEY_LOCKS = tuple(threading.RLock() for _ in range(_LOCK_COUNT))
_COMPUTATION_CONTEXT = threading.local()
_MISSING = object()


def _json_safe(value: Any, *, _seen: set[int] | None = None, _depth: int = 0) -> Any:
    """Detach values, rejecting ambiguous/non-finite JSON before keying or writing."""
    if _depth > 64:
        raise ValueError("Analysis cache value exceeds 64 nested levels")
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Analysis cache values must not contain NaN or infinity")
        return float(value)
    if isinstance(value, os.PathLike):
        path = os.fspath(value)
        if not isinstance(path, str):
            raise TypeError("Analysis cache paths must be text paths")
        return path

    seen = set() if _seen is None else _seen
    identity = id(value)
    if identity in seen:
        raise ValueError("Analysis cache values must not contain circular references")
    seen.add(identity)
    try:
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise TypeError("Analysis cache mapping keys must be strings")
            return {
                key: _json_safe(item, _seen=seen, _depth=_depth + 1)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [_json_safe(item, _seen=seen, _depth=_depth + 1) for item in value]

        # NumPy's own conversion produces ordinary nested lists/scalars.  Use the
        # protocol only for NumPy values; do not import NumPy (or run arbitrary
        # custom object conversion methods) as part of cache initialization.
        module = type(value).__module__
        if module == "numpy" or module.startswith("numpy."):
            convert = getattr(value, "tolist", None)
            if not callable(convert):
                convert = getattr(value, "item", None)
            if callable(convert):
                converted = convert()
                if converted is value:
                    raise TypeError("NumPy value cannot be represented as plain JSON")
                return _json_safe(converted, _seen=seen, _depth=_depth + 1)
        raise TypeError(f"Unsupported analysis cache value: {type(value).__name__}")
    finally:
        seen.remove(identity)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def file_identity(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Hash the bytes of one regular file; path/mtime are deliberately not identity.

    File symlinks are read through to their target.  Directories are not model
    identities: callers must identify the exact configuration and model assets
    they actually use, or provide independently pinned asset hashes.
    """
    source = Path(path)
    mode = source.stat().st_mode
    if stat.S_ISDIR(mode):
        raise IsADirectoryError(f"file_identity requires a regular file: {source}")
    if not stat.S_ISREG(mode):
        raise ValueError(f"file_identity requires a regular file: {source}")
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as handle:
        # fstat describes the opened file, avoiding a path replacement between
        # the regular-file check and the content read.
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError(f"file_identity requires a regular file: {source}")
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return {"kind": "file", "sha256": digest.hexdigest(), "size_bytes": size}


def array_identity(array: Any, sample_rate: int = 16000) -> dict[str, Any]:
    """Hash a NumPy-compatible array's C-order bytes and interpretation metadata.

    The caller owns the array and must not mutate it while this function runs.
    Strides, addresses, and C/F contiguity are not part of an audio observation's
    identity.  Shape, dtype (including byte order), sample rate, and bytes are.
    """
    if isinstance(sample_rate, bool):
        raise ValueError("sample_rate must be a positive integer")
    try:
        rate = operator.index(sample_rate)
    except TypeError as exc:
        raise TypeError("sample_rate must be a positive integer") from exc
    if rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    try:
        dtype = array.dtype
        shape = [operator.index(dimension) for dimension in array.shape]
        itemsize = operator.index(dtype.itemsize)
    except (AttributeError, TypeError) as exc:
        raise TypeError("array_identity requires an array with shape, dtype and tobytes") from exc
    if any(dimension < 0 for dimension in shape) or itemsize <= 0:
        raise ValueError("Array shape and dtype item size must be valid")
    if bool(getattr(dtype, "hasobject", False)):
        raise TypeError("Object arrays do not have stable content identities")
    # Structured dtypes can have identical dtype.str but different field meaning.
    if getattr(dtype, "fields", None) is not None:
        raise TypeError("Structured arrays are not supported audio inputs")
    try:
        content = array.tobytes(order="C")
    except (AttributeError, TypeError) as exc:
        raise TypeError("array_identity requires C-order tobytes support") from exc
    if not isinstance(content, bytes):
        raise TypeError("array.tobytes must return bytes")
    expected = math.prod(shape) * itemsize
    if len(content) != expected:
        raise ValueError("Array content length does not match its shape and dtype")
    return {
        "kind": "array",
        "sha256": hashlib.sha256(content).hexdigest(),
        "shape": shape,
        "dtype": str(getattr(dtype, "str", dtype)),
        "sample_rate": rate,
        "size_bytes": len(content),
    }


@contextmanager
def _disk_lock(path: Path) -> Iterator[bool]:
    """Acquire one of a bounded set of process locks; a crash releases the lock.

    Cache storage/locking failures must not turn successful inference into a
    playback failure.  A caller that cannot acquire this lock may read an atomic
    entry and compute a miss, but must not publish its computed value.
    """
    handle = None
    locked = False
    unlock = None
    try:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a+b")
            if os.fstat(handle.fileno()).st_size == 0:
                handle.write(b"\0")
                handle.flush()
            if os.name == "nt":
                import msvcrt

                def attempt() -> None:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

                def release() -> None:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                def attempt() -> None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                def release() -> None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

            deadline = time.monotonic() + _LOCK_WAIT_SECONDS
            while True:
                try:
                    attempt()
                    locked = True
                    unlock = release
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.01)
        except (OSError, ImportError):
            # Unsupported/read-only cache volumes retain normal inference.
            pass
        yield locked
    finally:
        if locked and unlock is not None:
            try:
                unlock()
            except OSError:
                pass
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass


class AnalysisCache:
    """Disk-backed cache for pure inference functions, independent of actor state.

    ``namespace`` must identify model bytes/configuration and source preprocessing
    policy.  ``inputs`` must identify all operation-specific immutable inputs.
    Values are always detached JSON values, including on a computed cache miss.
    No tensor, audio buffer, result, or per-key lock dictionary is retained in RAM.
    """

    def __init__(self, root: str | os.PathLike[str], namespace: Any):
        self.root = Path(root).expanduser().resolve(strict=False)
        self._namespace = _canonical(_json_safe(namespace))
        self._root_key = os.path.normcase(str(self.root)).encode("utf-8")
        self._stats_lock = threading.Lock()
        self._counts = {
            "hits": 0,
            "misses": 0,
            "computations": 0,
            "compute_failures": 0,
            "conversion_failures": 0,
            "corrupt_reads": 0,
            "read_failures": 0,
            "writes": 0,
            "write_failures": 0,
            "oversized_results": 0,
            "lock_bypasses": 0,
            "nested_bypasses": 0,
        }

    def _count(self, name: str) -> None:
        with self._stats_lock:
            self._counts[name] += 1

    def stats(self) -> dict[str, int]:
        """Return an independent snapshot of counters, never mutable internal state."""
        with self._stats_lock:
            return dict(self._counts)

    def _read(self, path: Path, key: str) -> Any:
        try:
            with path.open("rb") as handle:
                content = handle.read(_MAX_ENTRY_BYTES + 1)
        except FileNotFoundError:
            return _MISSING
        except OSError:
            self._count("read_failures")
            return _MISSING
        try:
            if len(content) > _MAX_ENTRY_BYTES:
                raise ValueError("Oversized cache record")
            record = json.loads(content)
            if not isinstance(record, dict) or record.get("format") != _FORMAT:
                raise ValueError("Unknown cache format")
            if record.get("key") != key or "value" not in record:
                raise ValueError("Cache key mismatch")
            result = record["value"]
            encoded = _canonical(result)
            if hashlib.sha256(encoded).hexdigest() != record.get("value_sha256"):
                raise ValueError("Cache value checksum mismatch")
            return result
        except (ValueError, TypeError, UnicodeError, RecursionError):
            self._count("corrupt_reads")
            return _MISSING

    def _write(self, path: Path, key: str, value: Any, encoded: bytes) -> None:
        record = _canonical({
            "format": _FORMAT,
            "key": key,
            "value_sha256": hashlib.sha256(encoded).hexdigest(),
            "value": value,
        })
        if len(record) > _MAX_ENTRY_BYTES:
            self._count("oversized_results")
            return
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=".analysis-", suffix=".tmp", delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(record)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            self._count("writes")
        except OSError:
            self._count("write_failures")
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def _compute_value(self, compute: Callable[[], Any]) -> tuple[Any, bytes]:
        self._count("computations")
        previous_depth = getattr(_COMPUTATION_CONTEXT, "depth", 0)
        _COMPUTATION_CONTEXT.depth = previous_depth + 1
        try:
            computed = compute()
        except Exception:
            self._count("compute_failures")
            raise
        finally:
            _COMPUTATION_CONTEXT.depth = previous_depth
        try:
            value = _json_safe(computed)
            return value, _canonical(value)
        except (ValueError, TypeError, UnicodeError, RecursionError):
            self._count("conversion_failures")
            raise

    def get_or_compute(self, operation: str, inputs: Any, compute: Callable[[], Any]) -> Any:
        """Return cached inference or invoke ``compute()`` exactly once in this call.

        Successful return values have identical JSON shapes on hits and misses.
        ``compute`` exceptions propagate unchanged and are never cached.  Invalid
        inputs/results raise TypeError/ValueError; unavailable cache storage only
        makes the result uncached.  A lock timeout also computes without writing.
        """
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError("Analysis cache operation must be a nonempty string")
        if not callable(compute):
            raise TypeError("Analysis cache compute must be callable")
        input_bytes = _canonical(_json_safe(inputs))
        key = hashlib.sha256(
            _FORMAT.encode("ascii") + b"\n" + self._namespace + b"\n"
            + _canonical(operation) + b"\n" + input_bytes
        ).hexdigest()
        path = self.root / "v1" / key[:2] / (key + ".json")
        if getattr(_COMPUTATION_CONTEXT, "depth", 0):
            # An outer cached computation owns a stripe lock.  A nested operation
            # may read a complete entry, but must not acquire another stripe or
            # OS byte lock: stripe collisions and opposite lock order can deadlock.
            # Nested misses remain useful inference and return without publishing.
            value = self._read(path, key)
            if value is not _MISSING:
                self._count("hits")
                return value
            self._count("nested_bypasses")
            self._count("misses")
            value, _ = self._compute_value(compute)
            return value
        stripe = hashlib.sha256(self._root_key + key.encode("ascii")).digest()[0]
        with _KEY_LOCKS[stripe]:
            # Atomic replace makes this optimistic read safe without a disk lock.
            value = self._read(path, key)
            if value is not _MISSING:
                self._count("hits")
                return value
            with _disk_lock(self.root / ".locks" / (key[:2] + ".lock")) as locked:
                if not locked:
                    self._count("lock_bypasses")
                # Another process may have completed while this call waited.
                value = self._read(path, key)
                if value is not _MISSING:
                    self._count("hits")
                    return value
                self._count("misses")
                value, encoded = self._compute_value(compute)
                if locked:
                    self._write(path, key, value, encoded)
                return value
