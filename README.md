# KevFLIX

A separate TV player for local movies, with PC-assisted English dubbing and optional upscaling.

**Development version 0.4.40 — playback, installer and update acceptance remain untested.**

KevFLIX keeps the original files intact and uses the PC for source decoding, audio processing, scaling and encoding. A NAS can supply the source files. The TV receives a local-network HLS stream and decodes that final stream for display. The standard Plex installation is not modified.

## Included

- TizenBrew application module and an unsigned standalone Tizen launcher.
- LAN-only media delivery, one-use TV pairing codes and revocable device tokens.
- Plex libraries in the TV's left sidebar, source selection for Plex or local folders, genre/year/collection filters, sorting, and a right-side letter index.
- Title-only poster labels, watched/progress overlays, movie information, cast/crew and Plex resume positions. Video playback is the current scope; other library types can be browsed.
- Original audio selection, text subtitles, optional Plex progress synchronization and playback controls.
- On-demand source-language detection, translation to English, speaker assignments, speech separation and local speech generation.
- A single combined audio/video stream for dubbing. Playback buffers while the next segment is processed; instantaneous or sustained real-time performance is **not established**.
- Output ceiling up to **3840×2160**, optional Lanczos upscaling, NVIDIA encoding, software-encoding option, and HDR/SDR controls.
- Optional Sem watch-along context and a staged integration recipe, without replacing Sem's existing voice.
- A Windows installer, PC control panel, visible TV pairing codes, Plex sign-in, private settings and startup with Windows enabled by default.
- Automatic desktop downloads from GitHub releases, installed on the next KevFLIX start; TV-interface updates applied between viewing sessions. Existing configuration and models are preserved.

## Installation paths

### Windows PC

Download **KevFLIX-Setup-0.4.40.exe** from [Releases](https://github.com/malov-dot/kevflix/releases). This unsigned development installer supplies the control panel and a private Python runtime. It starts in setup-only mode. Use PC settings to connect Plex, add folders, install video/dubbing components and generate the TV pairing code. Tools and models download separately; setup does not play media or run inference. See [installation](docs/INSTALL.md).

### TizenBrew GitHub module

Add this GitHub module:

    malov-dot/kevflix

This is the normal TizenBrew application-module route. TizenBrew fetches the module through its public CDN, so launching through this route can require internet access. Movies, dubbing audio and viewing context still stay on the LAN.

### Offline launch after installation

Download the unsigned WGT from the latest [development release](https://github.com/malov-dot/kevflix/releases) and use TizenBrew Installer or your normal Tizen signing/install workflow. The package needs the certificate/signature appropriate for the TV; this repository does not include signing credentials or a pre-signed package.

The standalone launcher contains its own interface and connects directly to the PC worker. After initial installation and model provisioning, playback does not require internet. Optional GitHub update checks can be disabled; a network failure retains the installed interface.

See [installation](docs/INSTALL.md), [architecture](docs/ARCHITECTURE.md), [updates](docs/UPDATES.md) and [validation boundaries](docs/VALIDATION.md).

## Privacy

This repository contains reusable source and generic installation files only. It does not include private configuration, credentials, movie filenames or metadata, network details, models, personal recordings, Sem profiles, logs or private recovery material. Keep relay-config.json, private-activation/, state/, models/ and setup-records/ out of version control.

## Dependencies and licensing

See [THIRD-PARTY.md](THIRD-PARTY.md) for model/runtime provenance. Upstream names and licenses remain unchanged. Models download separately. The Windows installer bundles a pinned Python runtime and its notices. A license for new KevFLIX source has not been selected; package metadata uses UNLICENSED.
