# Validation status

**Not run. This is an unvalidated development release.**

File construction and dependency installation do not establish playback correctness. No tests were added or run during this work. No model, decoder, encoder, TV player or generated-media preview was executed.

Deferred acceptance includes:

- TV installation, remote navigation, lifecycle and offline launch.
- Pairing/revocation, LAN-only URLs, path boundaries and stale session handling.
- Direct library browsing, poster loading and optional Plex resume/progress behavior.
- SD/HD/4K decoding and encoding, GPU admission, source frame rates and aspect ratios.
- HEVC HDR signaling, SDR tone mapping and Samsung HLS support.
- Dubbing meaning, speaker stability, voice quality, separation and timing.
- Seek/toggle/cancel/review transitions and subsequent playback.
- Segment timestamps, native HLS timeline origin, subtitle alignment and long sessions.
- Processing speed, startup delay, rebuffering, cache size, CPU/RAM/VRAM and shared NAS I/O.
- GitHub update/rollback behavior and failure during a viewing session.
- Sem observation timing, microphone/voice continuity and owner acceptance.

Known scope limits: local movie files first; text subtitles only; no bitmap subtitle OCR; no universal streaming-service/DRM integration; no neural super-resolution or invented actor voice-cloning claim; no zero-latency dubbing claim.
