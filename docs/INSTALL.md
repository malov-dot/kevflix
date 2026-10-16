# Install KevFLIX

## Current status

The source and packages are development artifacts. No runtime tests, model inference, encoder runs, TV installation or playback acceptance have been performed. Do not assume a packaged file has been validated on hardware.

## Windows installer

Download KevFLIX-Setup-0.4.2.exe from Releases. It installs under %LOCALAPPDATA%/Programs/KevFLIX with a private Python runtime. Startup with Windows is checked by default. The desktop shortcut opens the PC control panel already authenticated, with a large six-digit pairing code and a New pairing code button.

Connect Plex, select a local server, and add movie folders. Show Plex library paths gives the NAS paths to map to their corresponding PC-accessible folders. Verify the worker URL is the PC address reachable by the TV, particularly if there are multiple network adapters. No router forwarding is needed.

In Install processing components, install video tools for playback; install dependencies and download models for dubbing after reviewing upstream terms. Gated models need your accepted Hugging Face access and HF_TOKEN in the PC environment. Downloads do not run inference. Progress is recorded in state/component-install.log.

Playback starts disabled. When ready to use it, enable Allow movie playback and processing after setup, save, select Close KevFLIX worker, and reopen the desktop shortcut. Opening the panel or downloading models does not authorize an automatic playback test.

Windows removal unregisters startup and shortcuts while retaining app/config/model files for recovery. The original media, Plex and Sem are separate.

## PC setup from source

Use Windows x64, Python 3.11 and a recent NVIDIA driver for hardware encoding.

    .\Setup-Worker.ps1
    .\Setup-FFmpeg.ps1
    .\.venv\Scripts\python.exe .\scripts\configure.py

Read the upstream model notices and accept any gated model access terms before provisioning:

    .\Setup-Worker.ps1 -DownloadModels -AcceptModelTerms

Configure relay-config.json locally:

- bind: the PC's private LAN address.
- public_url: that address and worker port, reachable from the TV.
- media_roots and tv.roots: folders the PC can read. UNC paths are preferable to user-session drive letters for network shares.
- device: cpu or cooperative-cuda for the dubbing models. GPU admission never unloads another application's models.
- tv.video_encoder: auto/nvenc for NVIDIA encoding, or software for CPU encoding.
- tv.max_height: 480, 720, 1080 or 2160. The player can change the limit per session.
- tv.upscale: optional Lanczos resize; it is not a trained detail-reconstruction model.
- tv.hdr_mode: auto to preserve the HDR signal through HEVC, or sdr for PC tone mapping. Hardware acceptance is outstanding.
- Optional plex.url/token: your local Plex server and a privately stored access token. The token stays on the PC.
- path_mappings: exact Plex NAS paths mapped to allowed PC paths, needed to open Plex metadata items from the filesystem.
- updates: enable/disable automatic TV-interface updates and select the repository branch.

Do not open the worker to the public internet. Permit its configured port only from the intended private network/TV. No firewall or router changes are applied by these scripts.

When an activation/testing window has been authorized, start Start-KevFLIX.cmd. On the PC dashboard, enter the generated API key and choose Pair a TV. Open KevFLIX on the TV, enter the PC's worker URL, then enter the one-use pairing code. Revoke a paired TV from the PC dashboard if needed.

## TV installation

Target API baseline: Tizen 6.5 or newer. TizenBrew integration was written against its documented application-module format and the v2.0.5 release; installation has not been exercised.

- TizenBrew GitHub module: add malov-dot/kevflix. This launch route fetches public module assets over the internet.
- Offline-capable launcher: install dist/KevFLIX-0.3.0-unsigned.wgt after signing/resigning through the supported TV installation workflow. TizenBrew Installer supports local WGT installation workflows; certificate requirements depend on the TV and installation method.

The standalone launcher subsequently loads its player interface from the local PC. It does not fetch media, translation or speech generation from a cloud service. It requires the PC worker to be running.

## Playback

TV Settings has Connect with: Plex or Local movie folders. Libraries from the selected local Plex server appear directly on the left. Choose a library, filter by genre/year/collection, sort, search or use the right-hand letter index when sorted A-Z. This reads Plex metadata; it does not copy the Plex client's pinned ordering or online streaming services. Non-video libraries can be browsed; video playback is the current scope.

Poster tiles show titles only underneath with watched/progress overlays. Movie details include synopsis, year, runtime, ratings, genres, cast/crew and media formats. Resume and Play from beginning are on that page. Per-movie PC processing controls are available before and during playback. Embedded text subtitle tracks are selected after opening playback; image subtitle formats are not currently rendered.

Browse Movie folders, or choose Plex library after configuring Plex access and path mappings. KevFLIX asks the PC to encode the source; it does not request Plex transcoding. Enable English dubbing in the player. The picture and English soundtrack are muxed together on the PC, and playback waits for complete segments.

Settings include output limit, Lanczos upscaling, HDR handling, audio tracks, text subtitles, commentary mode and optional Plex progress synchronization. Plex server administration remains in Plex's own local web interface. The optional progress setting uses Plex's normal API; no direct database edits occur.

## Sem

The sem/ directory contains a candidate generator and optional integration. The example config is generic. Stage against the current Sem source, compare continuity requirements, and obtain the required owner acceptance before activation. No Sem runtime is included here.

## Removal

Stop only the KevFLIX worker and remove the TV launcher/module. Its cache and pairing records are under its own state directory. Original media and the Plex installation remain separate. The discarded transcoder-wrapper adapter is not part of this release and must not be installed.
