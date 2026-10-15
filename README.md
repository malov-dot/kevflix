# KevFLIX

A separate TV player for local movies, with PC-assisted English dubbing and optional upscaling.

**Development version 0.3.2 — playback and model processing remain unvalidated.**

KevFLIX keeps the original files intact and uses the PC for source decoding, audio processing, scaling and encoding. A NAS can supply the source files. The TV receives a local-network HLS stream and decodes that final stream for display. The standard Plex installation is not modified.

## Included

- TizenBrew application module and an unsigned standalone Tizen launcher.
- LAN-only media delivery, one-use TV pairing codes and revocable device tokens.
- Folder browsing and optional Plex movie-library browsing with posters, search and existing resume positions.
- Original audio selection, text subtitles, optional Plex progress synchronization and playback controls.
- On-demand source-language detection, translation to English, speaker assignments, speech separation and local speech generation.
- A single combined audio/video stream for dubbing. Playback buffers while the next segment is processed; instantaneous or sustained real-time performance is **not established**.
- Output ceiling up to **3840×2160**, optional Lanczos upscaling, NVIDIA encoding, software-encoding option, and HDR/SDR controls.
- Optional Sem watch-along context and a staged integration recipe, without replacing Sem's existing voice.
- Automatic updates for the TV interface from this repository, applied between viewing sessions. Worker/dependency and launcher updates remain separate installation steps.

## Installation paths

### TizenBrew GitHub module

Add this GitHub module:

    malov-dot/kevflix

This is the normal TizenBrew application-module route. TizenBrew fetches the module through its public CDN, so launching through this route can require internet access. Movies, dubbing audio and viewing context still stay on the LAN.

### Offline launch after installation

Use the unsigned WGT in dist/ with TizenBrew Installer or your normal Tizen signing/install workflow. The package needs the certificate/signature appropriate for the TV; this repository does not include signing credentials or a pre-signed package.

The standalone launcher contains its own interface and connects directly to the PC worker. After initial installation and model provisioning, playback does not require internet. Optional GitHub update checks can be disabled; a network failure retains the installed interface.

See [installation](docs/INSTALL.md), [architecture](docs/ARCHITECTURE.md), [updates](docs/UPDATES.md) and [validation boundaries](docs/VALIDATION.md).

## Privacy

This repository contains reusable source and generic installation files only. It does not include private configuration, credentials, movie filenames or metadata, network details, models, personal recordings, Sem profiles, logs or private recovery material. Keep relay-config.json, private-activation/, state/, models/ and setup-records/ out of version control.

## Dependencies and licensing

See [THIRD-PARTY.md](THIRD-PARTY.md) for model/runtime provenance. Upstream names and licenses remain unchanged. No third-party model weights or runtime binaries are included. A license for new KevFLIX source has not been selected; package metadata uses UNLICENSED.

## Connection setup

Start-KevFLIX-Setup.cmd enables pairing without playback or inference. See [connection setup](docs/CONNECTION-SETUP.md). The launcher now shows connection progress and errors, and accepts a PC address without requiring the HTTP prefix or default port to be typed.

## TV navigation

The simplified sidebar, poster grid, movie details and on-screen pairing keypad use arrows, OK and Back. No numeric remote is required. See [remote controls](docs/REMOTE-CONTROLS.md).
