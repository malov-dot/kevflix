"""Explicit movie failure metadata; no inference/process side effects."""
from __future__ import annotations
KINDS = frozenset({"fatal", "resource", "recoverable", "cancelled"})

class MovieFault(RuntimeError):
    fault_kind = "fatal"
    def __init__(self, code, detail):
        self.fault_code, self.detail = str(code), str(detail)
        super().__init__(self.fault_code + ": " + self.detail)
    def record(self):
        return {"kind": self.fault_kind, "code": self.fault_code, "detail": self.detail}

class MovieConfigurationError(MovieFault):
    """Retrying unchanged configuration cannot repair it."""

class MovieResourceWait(MovieFault):
    """Explicit transient lease/VRAM hold."""
    fault_kind = "resource"

class MovieContentFallback(MovieFault):
    """Preserve original speech for cue uncertainty."""
    fault_kind = "recoverable"

class MovieCancelled(MovieFault):
    fault_kind = "cancelled"

def fault_record(error):
    """Read explicit metadata from local or separately loaded adapter errors."""
    kind, code = getattr(error, "fault_kind", None), getattr(error, "fault_code", None)
    if not isinstance(kind, str) or kind not in KINDS or not isinstance(code, str) or not code:
        return None
    return {"kind": kind, "code": code,
            "detail": str(getattr(error, "detail", str(error)))}

def from_record(record):
    """Bad worker failure envelopes fail terminally rather than retry forever."""
    if (not isinstance(record, dict) or not isinstance(record.get("kind"), str)
            or record["kind"] not in KINDS):
        return MovieConfigurationError("invalid_worker_failure", "Invalid worker failure envelope")
    code, detail = record.get("code"), record.get("detail")
    if not isinstance(code, str) or not code or not isinstance(detail, str):
        return MovieConfigurationError("invalid_worker_failure", "Worker failure code/detail missing")
    cls = {"fatal": MovieConfigurationError, "resource": MovieResourceWait,
           "recoverable": MovieContentFallback, "cancelled": MovieCancelled}[record["kind"]]
    return cls(code[:120], detail[:4000])

