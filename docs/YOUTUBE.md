# YouTube on the TV (stage 1)

KevFLIX can show the real YouTube TV page (youtube.com/tv) on the TV. The page runs in ONE background (headless) Chrome on the PC; its
screen is streamed to the KevFLIX app and the TV remote keys are sent back to the page. Nothing is shown on the PC: no window, no taskbar
entry, no focus change, no sound, and no operating-system keyboard input (keys go to the page through the DevTools protocol only).

It is OFF by default. Turn it on in `relay-config.json`:

```json
"youtube": { "enabled": true }
```

| Setting | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Master switch. Off: the TV shows no YouTube tile and every `/tv-api/youtube/*` call except `status` is refused (403). |
| `profile_dir` | `C:/KevFLIX-Runtime/youtube-profile` | The dedicated browser profile (never your own Chrome profile; keep it out of `%LOCALAPPDATA%`). |
| `chrome_path` | `""` | Empty = Google Chrome, else Microsoft Edge, from their standard install folders. |
| `idle_seconds` | `120` | Chrome is closed after this long with no TV watching or pressing keys. |
| `grace_seconds` | `5` | The screen capture stops this long after the last TV leaves. |
| `user_agent` / `user_agent_profile` | `""` / `tizen` | The TV user agent that makes youtube.com/tv draw its TV interface (`tizen` or `cobalt`). |
| `width`, `height`, `jpeg_quality`, `every_nth_frame`, `max_fps` | 1920, 1080, 65, 3, 20 | Picture size and cost. `every_nth_frame` 3 on a 60 Hz page is about 20 pictures per second. |
| `memory_limit_mb` | `3072` | Windows Job Object memory cap for the whole Chrome tree (0 = none). |

Chrome runs at BELOW_NORMAL priority inside a Windows Job Object (it dies with the KevFLIX server), with one renderer process, muted
audio, media-key capture, notifications and permission prompts disabled, and DevTools bound to 127.0.0.1.

Endpoints (all need the paired TV token): `POST /tv-api/youtube/open`, `GET /tv-api/youtube/status`, `POST /tv-api/youtube/key`
(`{"key": "Up|Down|Left|Right|Enter|Back|Play|Pause|PlayPause|FastForward|Rewind", "action": "down|up|press"}`),
`POST /tv-api/youtube/close`, and `GET /tv-api/youtube/screen?ticket=...` (multipart MJPEG; the ticket comes from `open` because an
`<img>` cannot send a token header). On the TV a LONG Back press (0.7 s or more) returns to the KevFLIX home; a short Back goes to YouTube.

Picture transport: the TV normally PULLS pictures with `GET /tv-api/youtube/frame?ticket=...&after=<counter>&ms=<ms>` (`open` answers
`frame_pull: true`). Each request returns the newest JPEG newer than `after` (header `X-Frame` = its counter) or `204` when the page did
not change for about 1.5 s; the TV asks again only after the picture arrived, so a slow Wi-Fi link cannot build a backlog (the multipart
push stream above, still served as the fallback, did: seconds of lag). `ms` is the TV's own download + decode time of its last picture; a
slow average makes the PC send a smaller, lighter picture (`PICTURE_LEVELS` in `relay/youtube_host.py`: the configured size, then 80 % and
about 67 %, with a lower JPEG quality) and a fast one for 8 s brings it back.

## Hand-off to the KevFLIX player (stage 3)

When you pick a video in the YouTube view, KevFLIX plays it in its OWN player instead of streaming the PC's picture: the PC downloads
the video with yt-dlp, runs it through the normal movie pipeline (original audio, or English dubbing, plus your saved quality / upscaling
settings) and the TV shows the usual player (sound, seek, pause, Back). When the video ends, or you press Back, the TV returns to the YouTube view.

How it works:

1. The PC-side Chrome navigates to a watch route (`https://www.youtube.com/tv#/watch?v=<id>`, the hash route TizenTube also reads). The host
   records the 11-character id, pauses the page's own `<video>` and blocks its `googlevideo.com/videoplayback` requests (so the PC does not
   decode or stream it in parallel), and keeps Chrome alive while the hand-off lasts (up to 12 hours). The TV learns of the pick from the
   reply to the OK key (within ~0.7 s) or from the 1-second status poll (`play` in `GET /tv-api/youtube/status`).
2. The TV opens its player, which calls `POST /tv-api/open {"youtube": "<id>"}` about once a second. While the PC downloads, the reply is
   `{"preparing": true, "youtube": {"state", "percent", "text"}}` and the player shows "Downloading from YouTube… NN%" on its normal buffering
   card. Back cancels the download (`POST /tv-api/youtube/cancel`). Once the file is staged the same call returns an ordinary session
   object (plus a `youtube` block) and the player continues exactly as for a library movie. Only the 11-character id is ever sent; a URL is never accepted.
3. `POST /tv-api/youtube/resume` (called when the player closes, ends, fails or is cancelled) stops any download, unblocks the page's media
   and sends Back to the page, then the TV reopens the picture stream. If the download or the session fails, the TV returns to the YouTube
   view and shows the reason; a viewer is never trapped.

Rules: videos longer than `max_minutes` are refused with a message on the TV; an upcoming or just-ended live stream is refused with its own
message; only public videos work (no cookies and no
sign-in; a sign-in wall is reported as "YouTube needs a sign-in"). English dubbing defaults ON unless yt-dlp reports the video's language as
English (an unknown language is dubbed; when the video has no language, the original-language automatic captions `xx-orig` name it), and stays switchable. The audio track is the original one (the format selector prefers the track yt-dlp
marks `original`, never a YouTube auto-dub). Quality, upscaling and encoder choices are your saved TV settings.

| Setting | Default | Meaning |
|---|---|---|
| `handoff` | `true` | `false`: picks are not intercepted; the picture stream keeps playing the video as in stage 1. |
| `media_dir` | `C:/KevFLIX-Runtime/youtube-media` | Download folder (`<id>/<id>.mkv` + `meta.json`; a stable path per video, so a replay is instant and cached dubbing is reused). Admitted as a media source internally while `youtube.enabled` is true; `media_roots` is not edited. Keep it out of `%LOCALAPPDATA%`. |
| `max_minutes` | `240` | Longest video accepted. |
| `max_cache_gb` | `30` | Cache size cap; the least recently used folders are deleted first (never one a session is playing). |
| `max_height` | `1080` | Tallest video stream downloaded (smaller downloads; the TV output is limited by your quality setting anyway). |
| `node_path` | `""` | Path of `node.exe` for yt-dlp's YouTube script solver; empty = `node` from PATH. |

## Fast start and live streams

A video that is not going to be dubbed (English, or any language when `stream_foreign` is on) **starts while it is still arriving**; a live
stream plays too. Nothing about the library films changes (30 s segments, the 60 s start lead, 2160p).

Ordinary video:

1. yt-dlp only **resolves** the stream addresses (H.264 + AAC files up to `max_height`, the original audio track first).
2. One hidden `ffmpeg -c copy` (no transcoding) remuxes them into `media_dir/<id>/<id>.mkv` while the player already reads that file.
   googlevideo serves any read longer than 10 MiB at about 1.5x real time (measured: 10 MiB ranges run at 30+ MB/s, one 20 MiB range at
   0.4 MB/s), so ffmpeg reads a loopback proxy (127.0.0.1, random path token, googlevideo hosts only) that asks YouTube for consecutive 10 MiB
   ranges. A 12-minute 1080p video (150 MB) arrives in about 3 s on a fast line.
3. The player's session uses the **YouTube profile**: `stream_segment_seconds` (4 s) segments, a start lead of `stream_start_seconds`
   (about 6 s) instead of 60 s, output at most `stream_max_height` (1080p) and `stream_maxrate` (6 Mbit/s) for a Wi-Fi TV, stereo AAC. The
   next segments are prepared `stream_ahead_seconds` ahead of the playhead, and a segment is only started once the growing file really
   contains it (a reader of the file's own clusters says how many seconds are complete). A seek inside the downloaded part is immediate;
   a seek past it waits and the TV shows "Waiting for the video to download that far (NN%)…".
4. When ffmpeg finishes, the file is the normal cached file (`meta.json`), exactly as before; cache size cap, eviction and the in-use
   guard are unchanged. Back or stopping the session stops the download, deletes the unfinished file and the session's short segments.
5. **Dubbing** needs the whole file. A video that defaults to dubbing (foreign or unknown language) is therefore still downloaded
   completely first and opens like a library film. If dubbing is turned on during a YouTube-profile session, it is refused with "still
   downloading (NN%)" until the file is complete, then the session switches to the library profile (30 s segments, the asked quality) at
   the same position through the usual restart. `ffmpeg` that cannot start reading, or a video yt-dlp offers only as HLS, falls back to the
   complete download too.

Live stream: yt-dlp resolves the live HLS; **one** ffmpeg per session turns it into a sliding HLS of `live_segment_seconds` (2 s) segments
in `cache/tv-live/<session>/` (H.264 + AAC stereo, NVENC when usable and else software, like every other session), served through the
signed `/tv-media/<id>/` URLs. The session status carries `live: true` and a growing `duration`; there is no seeking, no dubbing ("Dubbing is
not available for live streams.") and the quality/encoder choices the TV sends are ignored. Back, a stopped session, an idle session, the
12-hour expiry and closing the server all end the ffmpeg and delete the folder; the ffmpeg children are also tied to the server by a
Windows Job Object, so they cannot outlive a killed server.

Optional settings (`youtube.<key>` in relay-config.json; the defaults live in `relay/youtube_source.py` `STREAM_DEFAULTS`):

| Setting | Default | Meaning |
|---|---|---|
| `stream` | `true` | `false`: every video is downloaded completely first, as before. |
| `stream_foreign` | `false` | `true`: videos that default to dubbing stream too (dubbing then waits for the finished file). |
| `stream_segment_seconds` | `4` | Segment length of a YouTube-profile session (2 to 30). |
| `stream_start_seconds` | `6` | Lead the TV is asked to buffer before it starts. |
| `stream_ahead_seconds` | `48` | How far past the playhead segments are prepared. |
| `stream_max_height` | `1080` | Output height cap of the YouTube profile. |
| `stream_maxrate` / `stream_bufsize` | `6M` / `12M` | ffmpeg rate cap of the YouTube profile. |
| `open_wait_seconds` | `1.5` | The TV's open request waits this long for the answer to change (no waiting between its once-a-second asks). |
| `live` | `true` | `false`: live streams are refused with the old message. |
| `live_segment_seconds` / `live_list_size` / `live_start_segments` | `2` / `8` / `2` | Live segment length, segments kept in the playlist, segments needed before the TV may start. |
| `live_max_height` / `live_maxrate` / `live_bufsize` | `1080` / `6M` / `12M` | Live output cap. |

Install yt-dlp into the project's own Python environment (the one named by the top-level `python` setting) and keep Node.js installed:

```
<project python> -m pip install -U "yt-dlp[default]"
```

`[default]` brings the EJS solver package (`yt-dlp-ejs`) that yt-dlp needs together with a JavaScript runtime. YouTube changes its site often; when
downloads start failing with errors from yt-dlp, update it with the same command (nothing else in KevFLIX needs to change). Downloading
videos is subject to YouTube's Terms of Service; KevFLIX uses it only for videos you pick yourself on your own PC, for private viewing.

## Userscripts (ad blocking and other page add-ons)

`youtube.userscripts` is a list of JavaScript files that the PC-side Chrome runs in the YouTube TV page. By default (`youtube.userscripts_when: "ready"`)
each file is evaluated through the browser's debugging connection when the page's DOMContentLoaded fires, because scripts written for TizenBrew
(for example TizenTube) do nothing when they run at document start; `"start"` injects the file before the page's own scripts. A missing or
oversized (> 4 MB) file is skipped with a log line and never stops the page from opening.

Third-party scripts are NOT part of this repository. TizenTube (`@foxreis/tizentube`, GPL-3.0-only, https://github.com/reisxd/TizenTube) provides ad
blocking and SponsorBlock by rewriting the page's own data; download its `dist/userScript.js` from npm yourself, keep it with its LICENSE outside the
repository (the reference install uses `C:/KevFLIX-Runtime/youtube-mods/tizentube/`), and list its path in `youtube.userscripts`. Running a GPL
script privately on your own PC does not change KevFLIX's own licence; do not redistribute it inside a KevFLIX release.
