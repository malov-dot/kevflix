# Architecture

    NAS/local files → PC source decode → optional dubbing/scaling → PC video/audio mux → LAN → TV
                          ↑
                  Plex metadata (optional)

The TV is a separate client. It does not patch the Plex player, replace Plex executables, install a transcoder wrapper or write the Plex database directly. Optional progress synchronization uses the normal Plex API.

The worker:

1. Reads only configured media roots.
2. Creates a revocable paired-device session.
3. Processes bounded segments and prioritizes the current seek position.
4. Keeps speech/language/speaker uncertainty in a review state instead of presenting fabricated English.
5. Muxes matching picture and audio into HLS transport-stream segments.
6. Signs local media URLs and expires unused sessions.

Dubbing uses local multilingual ASR, translation, diarization, separation and English synthesis. The code reuses exact cached model revisions where available; original model directories are not modified. Models and recordings are excluded from source publication.

Output is capped at 4K. Upscaling is Lanczos resizing. PC NVENC admission checks free GPU memory and waits when unavailable; it never evicts Sem or other processes. Software mode is available but may be far below real time at high resolution. TV hardware, network speed, codecs, HDR behavior and source complexity determine actual playback.

GitHub is a code-distribution channel only. It receives no movie bytes, subtitles, playback history or Sem observations. The standalone TV launcher loads its interface from the local worker. The public TizenBrew module route can require internet for its bootstrap.

No installable or automatically updated application can carry a zero-risk guarantee. Isolation reduces the scope of failures; playback, recovery and resource behavior still require acceptance testing.
