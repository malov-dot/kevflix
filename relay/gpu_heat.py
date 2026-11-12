"""Temperature of the graphics card, for the dub's heat guard (relay/movie_pipeline/pipeline.py) and the background gate."""
from __future__ import annotations
import os
import subprocess


def gpu_temperature_c():
    """Temperature in C of the hottest NVIDIA GPU, or None when it cannot be read."""
    try:
        flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        result = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                                capture_output=True, timeout=5, check=True, **flags)
        values = [float(line.strip()) for line in result.stdout.decode("ascii", "replace").splitlines() if line.strip()]
        return max(values) if values else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class HeatGuard:
    """Hot at or above `limit`; stays hot until the card is back at or below `resume`. limit 0 switches the guard off.
    An unreadable temperature never blocks: a film someone is watching is not stopped by a missing sensor reading
    (the background gate of F27 makes the opposite choice on purpose)."""

    def __init__(self, limit=85.0, resume=78.0, every=10.0, read=gpu_temperature_c):
        self.limit, self.resume, self.every, self.read = float(limit), float(resume), float(every), read
        self.hot, self.value, self._at = False, None, None

    def block(self, now):
        """A hold record ({obtainable: False, code: 'gpu_hot', detail, free_mib: None}) while hot, else None."""
        if self.limit <= 0:
            self.hot = False
            return None
        if self._at is None or now - self._at >= self.every:
            self._at, self.value = now, self.read()
        value = self.value
        if value is None:
            self.hot = False
            return None
        self.hot = value > self.resume if self.hot else value >= self.limit
        if not self.hot:
            return None
        return {"obtainable": False, "code": "gpu_hot", "free_mib": None,
                "detail": "graphics card at %.0f C (limit %.0f C); dubbing resumes at %.0f C" % (value, self.limit, self.resume)}
