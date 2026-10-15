from __future__ import annotations

import math
import re

from .language import words


def vocal_band_energy(samples, rate):
    import numpy as np
    data = np.asarray(samples)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if len(data) < 128:
        return None
    n = 1024
    energies = []
    band = (np.fft.rfftfreq(n, 1 / rate) >= 200) & (np.fft.rfftfreq(n, 1 / rate) <= 4000)
    for pos in range(0, max(1, len(data) - n + 1), n // 2):
        frame = data[pos:pos + n]
        if len(frame) < n:
            frame = np.pad(frame, (0, n - len(frame)))
        power = abs(np.fft.rfft(frame * np.hanning(n))) ** 2
        energies.append(float(power[band].mean()))
    return max(1e-14, sum(energies) / len(energies))


def separation_report(config, original, bed, rate, cues, origin, heard_bed):
    issues, measurements = [], []
    for cue in cues:
        left = max(0, round((cue["start"] - origin) * rate))
        right = min(len(original), round((cue["end"] - origin) * rate))
        if right <= left:
            continue
        before = vocal_band_energy(original[left:right], rate)
        after = vocal_band_energy(bed[left:right], rate)
        if before is None or after is None or before <= 1e-12:
            issues.append({"kind": "separation_inconclusive", "cue": cue.get("id")})
            continue
        suppression = 10 * math.log10(before / after)
        measurements.append({"cue": cue.get("id"), "suppression_db": round(suppression, 2)})
        if suppression < config["min_dialogue_suppression_db"]:
            issues.append({"kind": "dialogue_attenuation_low", "cue": cue.get("id"),
                           "suppression_db": round(suppression, 2)})
    audible = [s for s in heard_bed if s.get("no_speech_probability", 1) < .5
               and s.get("log_probability", -100) > -1.2 and len(words(s["text"])) >= 2]
    if audible:
        source = set(words(" ".join(c["text"] for c in cues)))
        for segment in audible:
            recognized = set(words(segment["text"]))
            overlap = len(source & recognized) / max(1, len(recognized))
            issues.append({"kind": "speech_in_background", "heard": segment["text"],
                           "source_word_overlap": round(overlap, 3),
                           "likely_original_dialogue": overlap >= config["residual_word_overlap"]})
    if not measurements and cues:
        issues.append({"kind": "separation_inconclusive", "reason": "No measurable dialogue window"})
    return {"machine_checks_passed": not issues, "issues": issues,
            "dialogue_windows": measurements, "human_listening_acceptance": False,
            "scope": "Attenuation and residual-ASR checks; not a guarantee of soundtrack fidelity"}
