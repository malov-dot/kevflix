"""Reference trust verification and render cache for actor-tts beta (worker 2).

Contract: actor-tts INTERFACES (reference cache / trust); A's module, only this docstring line changed.
Dependencies: stdlib + numpy + soundfile. No network, no GPU, no bank writes.

Trust model
-----------
Only a ``TrustAdmission`` minted by the trust owner's ``TrustIssuer`` (secret
held by B via the supervisor) authorizes a reference. The admission is an
HMAC-SHA256 signed envelope over source/job/actor/reference sha256/reference
text sha256/ranges. A dict, a bool, or ``trusted=True`` never authorizes.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ADAPTER = _HERE.parent / "runtime-adapter"

try:  # worker 1 owns errors.py; fall back to a compatible class if absent.
    if (_ADAPTER / "errors.py").is_file() and str(_ADAPTER) not in sys.path:
        sys.path.insert(0, str(_ADAPTER))
    from errors import ActorUnavailable  # type: ignore
except Exception:  # pragma: no cover - depends on concurrent worker
    class ActorUnavailable(Exception):
        """Fallback compatible with runtime-adapter/errors.ActorUnavailable."""

        def __init__(self, reason, detail=None):
            self.reason = reason
            self.detail = detail
            super().__init__(f"{reason}: {detail}" if detail else reason)


REASONS = ("untrusted_actor", "reference_unverified", "reference_invalid")
ENVELOPE_STATUSES = ("verified", "refused", "unknown")
CACHE_SAMPLE_RATE = 48000
CACHE_CHANNELS = 1
_ADMISSION_VERSION = "actor-tts-trust-admission/v1"
_IDENTITY_VERSION = "actor-tts-render-identity/v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _unavailable(reason, detail=None):
    assert reason in REASONS, reason
    return ActorUnavailable(reason, detail)


def text_sha256(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def file_sha256(path):
    """Same algorithm as relay.media.file_hash (sha256 over raw bytes)."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _norm_ranges(ranges):
    out = []
    for item in ranges or ():
        start, end = item
        out.append((round(float(start), 6), round(float(end), 6)))
    return tuple(out)


# --------------------------------------------------------------------------
# Trust admission
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TrustAdmission:
    source_id: str
    job_id: str
    actor_id: str
    reference_sha256: str
    reference_text_sha256: str
    ranges: tuple
    issuer_id: str
    signature: str = field(repr=False)

    def payload(self):
        return _admission_payload(self.source_id, self.job_id, self.actor_id,
                                  self.reference_sha256, self.reference_text_sha256,
                                  self.ranges, self.issuer_id)


def _admission_payload(source_id, job_id, actor_id, reference_sha256,
                       reference_text_sha256, ranges, issuer_id):
    body = {
        "v": _ADMISSION_VERSION,
        "source_id": str(source_id),
        "job_id": str(job_id),
        "actor_id": str(actor_id),
        "reference_sha256": str(reference_sha256).lower(),
        "reference_text_sha256": str(reference_text_sha256).lower(),
        "ranges": [list(r) for r in _norm_ranges(ranges)],
        "issuer_id": str(issuer_id),
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


class TrustIssuer:
    """Owned by the trust owner (B via supervisor). Holds the HMAC secret."""

    def __init__(self, secret_bytes, issuer_id="trust-owner-b"):
        if not isinstance(secret_bytes, (bytes, bytearray)) or len(secret_bytes) < 16:
            raise ValueError("TrustIssuer secret must be >= 16 bytes")
        self._secret = bytes(secret_bytes)
        self.issuer_id = str(issuer_id)

    def __repr__(self):
        return f"TrustIssuer(issuer_id={self.issuer_id!r})"

    def _sign(self, payload):
        return hmac.new(self._secret, payload, hashlib.sha256).hexdigest()

    def mint(self, source_id, job_id, actor_id, reference_sha256,
             reference_text_sha256, ranges):
        ranges = _norm_ranges(ranges)
        payload = _admission_payload(source_id, job_id, actor_id, reference_sha256,
                                     reference_text_sha256, ranges, self.issuer_id)
        return TrustAdmission(str(source_id), str(job_id), str(actor_id),
                              str(reference_sha256).lower(),
                              str(reference_text_sha256).lower(), ranges,
                              self.issuer_id, self._sign(payload))

    def verify(self, admission):
        if type(admission) is not TrustAdmission:
            return False
        if admission.issuer_id != self.issuer_id:
            return False
        try:
            expected = self._sign(admission.payload())
        except Exception:
            return False
        return hmac.compare_digest(expected, str(admission.signature))


def verify_admission(admission, issuer):
    """True only for an unmodified TrustAdmission minted by ``issuer``."""
    if not isinstance(issuer, TrustIssuer):
        return False
    return issuer.verify(admission)


# --------------------------------------------------------------------------
# Envelope adaptation (PROVISIONAL)
# --------------------------------------------------------------------------

def adapt_envelope(raw):
    """Map B's per-actor envelope into the normalized form used here.

    PROVISIONAL: B/C final schema is pending; these fields are provisional and
    NOT validated against any agreed schema. Expected raw fields:
    ``source_id, job_id, actor_id, reference_path, reference_sha256,
    reference_text, verified_ranges [[start, end], ...], status`` with status in
    {'verified', 'refused', 'unknown'}. Missing/None input maps to status
    'absent'; an unrecognised status maps to 'unknown'. The envelope alone never
    authorizes anything; a signed TrustAdmission is still required.
    """
    if not isinstance(raw, dict):
        return {"status": "absent", "_normalized": True}
    status = str(raw.get("status") or "unknown").strip().lower()
    if status not in ENVELOPE_STATUSES:
        status = "unknown"
    try:
        ranges = _norm_ranges(raw.get("verified_ranges"))
    except Exception:
        ranges = ()  # malformed -> reference_invalid later
    text = raw.get("reference_text")
    text = "" if text is None else str(text)
    return {
        "_normalized": True,
        "status": status,
        "source_id": None if raw.get("source_id") is None else str(raw["source_id"]),
        "job_id": None if raw.get("job_id") is None else str(raw["job_id"]),
        "actor_id": None if raw.get("actor_id") is None else str(raw["actor_id"]),
        "reference_path": raw.get("reference_path"),
        "reference_sha256": str(raw.get("reference_sha256") or "").lower(),
        "reference_text": text,
        "reference_text_sha256": text_sha256(text),
        "ranges": ranges,
    }


# --------------------------------------------------------------------------
# Reference verification
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class VerifiedReference:
    path: str
    sha256: str
    text: str
    text_sha256: str
    actor_id: str
    source_id: str
    job_id: str
    ranges: tuple
    seconds: float


def _voice_actor(voice):
    actor = voice.get("actor_id")
    if actor is None and isinstance(voice.get("trust"), dict):
        actor = voice["trust"].get("actor_id")
    return None if actor is None else str(actor)


def verify_reference(voice, envelope, admission, issuer, *, max_seconds=15,
                     range_tolerance=0.25):
    """Return VerifiedReference or raise ActorUnavailable(reason).

    reasons: untrusted_actor (absent/refused/unknown envelope, no signed
    admission), reference_unverified (bad signature or any source/job/actor/
    ref sha/text sha/ranges mismatch), reference_invalid (file hash mismatch,
    unreadable/overlong/silent wav, empty or inconsistent ranges).
    """
    if not isinstance(voice, dict):
        raise _unavailable("untrusted_actor", "voice is not a mapping")
    env = envelope if isinstance(envelope, dict) and envelope.get("_normalized") else adapt_envelope(envelope)
    if env.get("status") != "verified":
        raise _unavailable("untrusted_actor", f"envelope status {env.get('status')}")
    if type(admission) is not TrustAdmission:
        raise _unavailable("untrusted_actor", "no signed trust admission")
    if not verify_admission(admission, issuer):
        raise _unavailable("reference_unverified", "admission signature invalid")
    for name in ("source_id", "job_id", "actor_id", "reference_sha256", "reference_text_sha256"):
        if getattr(admission, name) != env.get(name):
            raise _unavailable("reference_unverified", f"admission {name} mismatch")
    if tuple(admission.ranges) != tuple(env.get("ranges") or ()):
        raise _unavailable("reference_unverified", "admission ranges mismatch")
    voice_actor = _voice_actor(voice)
    if voice_actor is not None and voice_actor != env["actor_id"]:
        raise _unavailable("reference_unverified", "voice actor mismatch")
    voice_text = voice.get("reference_text")
    voice_text = "" if voice_text is None else str(voice_text)
    if text_sha256(voice_text) != env["reference_text_sha256"]:
        raise _unavailable("reference_unverified", "reference text mismatch")

    path = voice.get("reference")
    if not path or not Path(path).is_file():
        raise _unavailable("reference_invalid", "reference file missing")
    try:
        actual = file_sha256(path)
    except OSError as exc:
        raise _unavailable("reference_invalid", f"unreadable: {exc}")
    if actual != str(voice.get("reference_sha256") or "").lower():
        raise _unavailable("reference_invalid", "file hash != voice reference_sha256")
    if actual != env["reference_sha256"]:
        raise _unavailable("reference_invalid", "file hash != envelope reference_sha256")

    try:
        import numpy as np
        import soundfile as sf
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise _unavailable("reference_invalid", f"wav unreadable: {exc}")
    if rate <= 0 or data.size == 0:
        raise _unavailable("reference_invalid", "empty wav")
    seconds = data.shape[0] / float(rate)
    if seconds > float(max_seconds):
        raise _unavailable("reference_invalid", f"too long {seconds:.2f}s > {max_seconds}s")
    if not np.all(np.isfinite(data)) or float(np.max(np.abs(data))) < 1e-5:
        raise _unavailable("reference_invalid", "silent or non-finite reference")
    ranges = env.get("ranges")
    if not ranges:
        raise _unavailable("reference_invalid", "no verified ranges")
    if any(end <= start for start, end in ranges):
        raise _unavailable("reference_invalid", "degenerate range")
    # relay/actor_reference.make_reference joins ranges with 0.12 s gaps.
    expected = sum(end - start for start, end in ranges) + 0.12 * (len(ranges) - 1)
    if abs(expected - seconds) > float(range_tolerance):
        raise _unavailable("reference_invalid",
                           f"ranges total {expected:.3f}s inconsistent with wav {seconds:.3f}s")
    return VerifiedReference(str(path), actual, voice_text, env["reference_text_sha256"],
                             env["actor_id"], env["source_id"], env["job_id"],
                             tuple(ranges), seconds)


def make_reference_gate(envelopes_by_actor, admissions_by_actor, issuer, *, max_seconds=15):
    """Return gate(voice) -> VerifiedReference for the runtime renderer.

    The actor id is taken from voice['actor_id'] or voice['trust']['actor_id'];
    the envelope and admission come ONLY from the supervisor-provided maps
    (anything else inside voice['trust'] is ignored, never trusted).
    """
    envelopes = dict(envelopes_by_actor or {})
    admissions = dict(admissions_by_actor or {})

    def gate(voice):
        if not isinstance(voice, dict):
            raise _unavailable("untrusted_actor", "voice is not a mapping")
        actor = _voice_actor(voice)
        if actor is None:
            raise _unavailable("untrusted_actor", "voice has no actor_id")
        return verify_reference(voice, envelopes.get(actor), admissions.get(actor),
                                issuer, max_seconds=max_seconds)

    return gate


# --------------------------------------------------------------------------
# Render cache
# --------------------------------------------------------------------------

IDENTITY_FIELDS = ("source_id", "job_id", "actor_id", "model_id", "model_manifest_sha256",
                   "reference_sha256", "reference_text_sha256", "settings", "text",
                   "context_key", "epoch")


def cache_identity(source_id, job_id, actor_id, model_id, model_manifest_sha256,
                   reference_sha256, reference_text_sha256, settings, text,
                   context_key, epoch):
    """Deterministic sha256 hex key over every input that affects a render."""
    body = {
        "v": _IDENTITY_VERSION,
        "source_id": str(source_id), "job_id": str(job_id), "actor_id": str(actor_id),
        "model_id": str(model_id), "model_manifest_sha256": str(model_manifest_sha256).lower(),
        "reference_sha256": str(reference_sha256).lower(),
        "reference_text_sha256": str(reference_text_sha256).lower(),
        "settings": settings or {}, "text": str(text),
        "context_key": None if context_key is None else str(context_key),
        "epoch": int(epoch),
    }
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


cache_key = cache_identity  # alias named in INTERFACES.md section 3


class StaleEpoch(Exception):
    """store() was begun under an epoch older than the cache's current epoch."""


class RenderCache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.entries = self.directory / "entries"
        self.quarantine_dir = self.directory / "quarantine"
        self.entries.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self._epoch_file = self.directory / "EPOCH"
        self._lock = threading.Lock()

    # epoch ------------------------------------------------------------
    def current_epoch(self):
        try:
            return int(self._epoch_file.read_text(encoding="ascii").strip() or 0)
        except (OSError, ValueError):
            return 0

    def bump_epoch(self):
        with self._lock:
            new = self.current_epoch() + 1
            self._atomic_write(self._epoch_file, str(new).encode("ascii"))
            return new

    # helpers ----------------------------------------------------------
    @staticmethod
    def _atomic_write(path, data):
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        with open(tmp, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)

    def _paths(self, key):
        return self.entries / f"{key}.wav", self.entries / f"{key}.json"

    def _quarantine(self, key, why):
        stamp = time.strftime("%Y%m%dT%H%M%S") + f"-{time.perf_counter_ns() % 10**9}"
        for src in self._paths(key):
            if src.exists():
                try:
                    os.replace(src, self.quarantine_dir / f"{key}-{stamp}{src.suffix}")
                except OSError:
                    pass
        try:
            (self.quarantine_dir / f"{key}-{stamp}.why").write_text(why, encoding="utf-8")
        except OSError:
            pass

    @staticmethod
    def _geometry_ok(path):
        try:
            import soundfile as sf
            info = sf.info(str(path))
        except Exception:
            return False
        return (info.samplerate == CACHE_SAMPLE_RATE and info.channels == CACHE_CHANNELS
                and info.frames > 0)

    # API --------------------------------------------------------------
    def lookup(self, key, identity=None):
        """Return (wav_path, receipt) on a verified hit, else None.

        A present-but-invalid entry (identity/sha/geometry/epoch mismatch) is
        quarantined. If ``identity`` is given, it must hash to ``key``.
        """
        if not isinstance(key, str) or not _HEX64.match(key):
            return None
        if identity is not None:
            try:
                if cache_identity(**identity) != key:
                    return None
            except TypeError:
                return None
        wav, meta = self._paths(key)
        if not meta.exists() and not wav.exists():
            return None
        try:
            record = json.loads(meta.read_text(encoding="utf-8"))
            stored_identity = record["identity"]
            receipt = record["receipt"]
            wav_sha = record["wav_sha256"]
        except Exception:
            self._quarantine(key, "unreadable receipt")
            return None
        try:
            if record.get("cache_key") != key or cache_identity(**stored_identity) != key:
                self._quarantine(key, "identity mismatch")
                return None
        except TypeError:
            self._quarantine(key, "identity malformed")
            return None
        if int(stored_identity.get("epoch", -1)) != self.current_epoch():
            self._quarantine(key, "stale epoch")
            return None
        if (not wav.exists() or file_sha256(wav) != wav_sha
                or receipt.get("sha256", wav_sha) != wav_sha):
            self._quarantine(key, "wav sha mismatch")
            return None
        if not self._geometry_ok(wav):
            self._quarantine(key, "geometry not 48000 Hz mono")
            return None
        out = dict(receipt)
        out["cache_hit"] = True
        return wav, out

    def store(self, key, identity, wav_path, receipt, *, epoch):
        """Atomically store a rendered wav. Raises StaleEpoch / ValueError."""
        if int(epoch) != self.current_epoch():
            raise StaleEpoch(f"store epoch {epoch} != current {self.current_epoch()}")
        if not isinstance(key, str) or not _HEX64.match(key):
            raise ValueError("bad cache key")
        identity = dict(identity)
        if int(identity.get("epoch", -1)) != int(epoch):
            raise ValueError("identity epoch != store epoch")
        if cache_identity(**identity) != key:
            raise ValueError("identity does not hash to key")
        if not self._geometry_ok(wav_path):
            raise ValueError("wav must be 48000 Hz mono")
        wav_sha = file_sha256(wav_path)
        if receipt.get("sha256") not in (None, wav_sha):
            raise ValueError("receipt sha256 != wav sha256")
        wav, meta = self._paths(key)
        tmp_wav = wav.with_name(f".{wav.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        shutil.copyfile(wav_path, tmp_wav)
        if file_sha256(tmp_wav) != wav_sha:
            tmp_wav.unlink(missing_ok=True)
            raise ValueError("wav changed during store")
        record = {"cache_key": key, "identity": identity, "wav_sha256": wav_sha,
                  "receipt": dict(receipt, sha256=wav_sha), "stored_epoch": int(epoch)}
        data = json.dumps(record, sort_keys=True, default=str).encode("utf-8")
        with self._lock:
            if int(epoch) != self.current_epoch():  # bumped while rendering/copying
                tmp_wav.unlink(missing_ok=True)
                raise StaleEpoch(f"epoch bumped to {self.current_epoch()} during store")
            os.replace(tmp_wav, wav)
            self._atomic_write(meta, data)  # receipt is the commit marker
        return wav
