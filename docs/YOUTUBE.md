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
