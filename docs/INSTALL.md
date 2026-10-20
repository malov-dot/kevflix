# Install KevFLIX

## Current status

The source and packages are development artifacts. No runtime tests, model inference, encoder runs, TV installation or playback acceptance have been performed. Do not assume a packaged file has been validated on hardware.

## Windows installer

`KevFLIX-Setup-<version>.exe` installs the app code and a private Python runtime under %LOCALAPPDATA%/Programs/KevFLIX. Startup with Windows is checked by default for a new install. An update keeps the choice you made earlier.

### Start at login and the tray icon

"Start with Windows" (a checkbox in the installer, a switch in the app's settings) creates two per-user Windows scheduled tasks that run when you log in, with your normal privileges and no time limit:
- `KevFLIX Server` starts the PC worker about 15 seconds after login (it is restarted up to three times if it stops at once).
- `KevFLIX Tray` shows the KevFLIX icon in the notification area about 10 seconds after login: an orange K while the worker answers, a grey K when it does not. Right-click for Open KevFLIX PC, Start server, Stop server and Exit tray icon; double-click opens KevFLIX PC. Closing the icon does not stop the worker. Windows may place a new icon under the ^ overflow arrow: drag it onto the taskbar to keep it visible. The tray log is `<profile>/logs/tray.log`.
Turning the switch off, or uninstalling, removes both tasks and the older `KevFLIX` entry under HKCU\...\Run that earlier builds wrote (on one PC Windows read that entry at login but never launched it). Check the tasks with `schtasks /Query /TN "KevFLIX Server"`.

### Your profile: settings and data that are never replaced

Your settings and data live in the KevFLIX profile, separate from the versioned app folder. The default location is %LOCALAPPDATA%/KevFLIX/profile. To use another folder, set KEVFLIX_PROFILE, or write {"profile_root": "<folder>"} to %LOCALAPPDATA%/KevFLIX/profile-location.json.

The profile holds:
- relay-config.json: Plex account and server, movie folders, UI settings, LAN bind/port, the Windows-startup preference;
- state/: library, sessions, resume and watched positions, TV pairing, actor identities;
- cache/: completed dubs;
- actor-bank/;
- desktop-updates/;
- logs/;
- runtime/.

How installs and updates treat it:
- Install and update never reset or overwrite it, and never rebuild it from config.example.json.
- An older install's relay-config.json is copied into the profile once; the original is kept. Existing state and dubs keep their location.
- Only a brand-new PC gets new settings with new local secrets.
- An update refreshes only the app-owned code pins in your settings (the Vox adapter hashes and the bundled lease client). Nothing else changes.

A completed dub is reused only when the media, the engine, the processing configuration and the output format all match. Otherwise it is processed again; nothing is served stale.

### Vox actor voice engine (dubbed voices)

The installer has a Vox step:
- Vox voice model file: voxcpm2-actor-q4_k.gguf.
- Vox native runtime folder: audiocpp.dll and its approved CUDA, ggml and MSVC DLLs.
- Both fields are pre-filled from an existing configured install.

The 1.7 GB model and the CUDA DLLs are not embedded in the installer. They are provisioned from the source you choose:
- Every file's sha256 is verified against actor-tts/runtime-adapter/approved.py.
- The files are copied into the profile (models/, tools/vox-native/), or used in place if you tick that option.
- The per-machine actor_runtime_manifest and the Vox settings are written into your settings, and Vox becomes the clone engine. Nothing else changes.
- A missing or mismatching file is reported by name, and Setup does not claim success when Vox is unusable.
- KevFLIX never downloads these files and never takes them from Sem's folders.

Unattended alternatives:
- KevFLIX Setup: /vox-model <file or folder> /vox-runtime <folder> [/vox-in-place]
- Setup-Worker.ps1: -VoxModelSource <...> -VoxRuntimeSource <...> [-VoxReferenceInPlace]

### Other components

In KevFLIX PC, under Install processing components:
- Install the video tools (FFmpeg).
- Install the Python dependencies: Setup-Worker.ps1 creates .venv from requirements.txt, with torch/torchaudio from the cu128 index.
- Download the models after reviewing their upstream terms. Gated models need your accepted Hugging Face access and HF_TOKEN.

These steps download but run no inference.

### Optional Qwen3-8B translator (translation_engine)

Movie dubbing translates the recognised speech to English with M2M100 by default. Two keys in relay-config.json select Qwen3-8B (Qwen team, Apache-2.0) instead:

    "translation_engine": "qwen3",
    "translation_llm_model": "D:/models/qwen3-8b"

- Without these keys (every existing settings file) or with "translation_engine": "m2m100", nothing changes. Any other engine value is refused at start.
- translation_llm_model is a local folder, absolute or relative to relay-config.json. It must exist when the engine is "qwen3". KevFLIX never downloads or ships these weights: copy the Qwen3-8B folder (config.json, generation_config.json, tokenizer files, model.safetensors.index.json and the five bf16 model-0000x-of-00005.safetensors files) there yourself, with its LICENSE.
- Size: about 16.4 GB on disk and about 16.4 GB of GPU memory (bf16, CUDA only; there is no CPU fallback). The first worker start reads the whole folder once to identify it (on the order of a minute on an SSD; unchanged files are remembered afterwards, and the 600 s worker start limit covers it); the model itself loads at the first translation of a movie. Hugging Face's .cache folder inside the model folder is ignored.
- GPU limits are unchanged. The movie lease still keeps its 3072 MiB free-VRAM floor (gpu_lease.reserve_mib), and the analysis worker's share of the lease budget (budget minus the other GPU workers) must hold the translator. If it cannot, the start is held as a resource wait with the numbers in the message; free VRAM (for example by stopping Sem or other GPU programs) and retry.
- Check the folder without loading anything or using the network: `python scripts/provision_models.py --verify-translation-llm <folder>` (or `--verify-translation-llm --config <relay-config.json>`). The startup preflight performs the same file check when the engine is "qwen3" and then no longer requires the M2M100 folder.
- Switching the translator or the Qwen3 folder changes the engine fingerprint and the analysis-cache identity, so completed dubs and cached translations of the other translator are never reused.
- int8 weight-only mode: `"translation_quantization": "int8"` stores the decoder weights as 8-bit (about 9 GB of GPU memory instead of 16.4 GB) with translations judged equal in a blind comparison. The default stays bf16.

### Optional audio-event classifier (audio_event_model)

Singing, chanted prayer, coughs, sneezes, sighs and similar non-speech sounds keep their original audio instead of being dubbed (see docs/DUBBING-SETTINGS.md). The text and pitch rules always apply; one key adds the AudioSet classifier MIT/ast-finetuned-audioset-10-10-0.4593 (BSD-3-Clause, 346 MB):

    "audio_event_model": "D:/models/ast-audioset"

- The value is a local folder holding config.json, preprocessor_config.json and model.safetensors. KevFLIX never downloads or ships it. Fetch it once: `python -c "from huggingface_hub import snapshot_download as d; d('MIT/ast-finetuned-audioset-10-10-0.4593', local_dir=r'D:/models/ast-audioset', allow_patterns=['config.json','preprocessor_config.json','model.safetensors'])"`.
- Empty or absent (the default): singing is then detected from pitch alone, which finds fewer sung lines; everything else still applies.

#### Smaller GPU footprint: translation_quantization "int8"

A third key, used only with "translation_engine": "qwen3", trades a little translation fidelity for about 7 GB less GPU memory:

    "translation_quantization": "int8"

- Values: "none" (the default, also when the key is missing, null or empty: unchanged bf16 behaviour) or "int8". Any other value (for example "int4", "INT8", a number) is refused at start. The key is ignored when the engine is "m2m100".
- It uses the same local bf16 Qwen3-8B folder: nothing is downloaded, converted on disk or written back. While each weight is read, every decoder-layer linear (q/k/v/o and gate/up/down, 252 matrices) is stored as weight-only INT8 with one float32 scale per output row (symmetric, scale = max|w|/127, rounded to nearest-even). Embeddings, norms and lm_head stay bf16 and activations stay bf16. Each call rebuilds the bf16 weight of one matrix temporarily and frees it, so a full dequantized copy never exists. It is pure PyTorch: bitsandbytes and accelerate are not needed.
- Resident weights on the GPU for Qwen3-8B (exact, from the model's shapes): 9,444,108,544 bytes (9,007 MiB) instead of 16,381,470,720 bytes (15,623 MiB), 6,616 MiB less. The admission check needs about 11,162 MiB (9,007 resident plus 2,155 computed headroom for the KV cache, activations and one dequantized matrix at the largest batch the translator allows) instead of 17,159 MiB. The lease's 3072 MiB free-VRAM floor still applies on top.
- Translation speed: dequantization runs on every call, so expect somewhat slower generation than bf16 (measure it with tests/translation/measure_int8_vs_bf16.py). Output text can differ from bf16 on some sentences; that script compares both on the same clips. Greedy decoding, the prompt and batching are the same.
- int8 and bf16 never mix: the setting is part of the analysis-cache identity (basis qwen3-8b-int8-to-English instead of qwen3-8b-bf16-to-English), of the engine fingerprint (only when it is "int8"; "none" keeps the existing fingerprints) and of the preflight/provision output. Switching it re-translates; completed dubs of the other precision are not reused.
- `python scripts/provision_models.py --verify-translation-llm --config <relay-config.json>` (or `--translation-quantization int8`) reports the GPU size of the selected mode without loading anything.

### Integrity, start and stop

**Integrity.** Every start verifies the installed app against its integrity manifest (app-manifest.json, created by the build). A changed, missing or unlisted code or asset file stops the start; there is no bypass. The manifest does not cover .venv, models, FFmpeg, the Vox engine files or your profile; they are provisioned and verified separately.

**Starting.** KevFLIX starts with Windows, from the KevFLIX PC shortcut, or with Start-KevFLIX.cmd. Start-KevFLIX-Setup.cmd starts setup-only mode.
- One app guard owns the app's lifetime. It registers the active instance in the profile and runs the server.
- The movie GPU lease is held inside the server. Only the GPU workers (Demucs, Vox) run in the lease's job.
- If Sem starts, or VRAM runs short, those workers stop. The server keeps playing cached, published and original audio.
- gpu_lease.sem_mode is "absent" when Sem is shut down for movies, and "running" (with gpu_lease.status_file) otherwise.

**Stopping.** Stop-KevFLIX.cmd (scripts/stop_worker.py) stops only the registered, validated KevFLIX instance. It reports "stopped", "not stopped" with the reason, or "no active instance". A stale entry is removed without signalling any process.

**Removal.** Windows removal stops KevFLIX, unregisters startup and the shortcut, and keeps the app files and your profile.

## Build (maintainers)

    python scripts/build_installer.py --source <app tree> --runtime <python-runtime.tar.gz> --output <dist> [--base-revision <40-hex base>]

**Checks; the build fails if any of these fails:**
- the pinned portable Python;
- one version in package.json, relay/__init__.py and tv/package.json;
- byte-identity of the bundled lease client, its tolerant reader and its five guard files, plus relay/actor_voice.py and relay/resource_budget.py;
- no development path in packaged text, apart from a reviewed exact allowlist of protection denylists.

**What the build generates:**
- actor_beta.vox.sources_sha256, written into the packaged default config;
- the integrity manifest.

**Build identity.** The identity is the manifest sha256 (status "unpublished local build"). --base-revision only records the publication base.

**Outputs.** Under the --output folder:
- KevFLIX-Setup-<version>.exe and its .sha256
- desktop-release.json: what is embedded and what is provisioned
- build-files.json: the packaged and excluded files, and any dev-path hits

## PC setup from source

A source tree has no integrity manifest, so the app guard refuses to start it. Build the installer and install it.

Do not open the worker to the public internet. Permit its configured port only from the intended private network/TV (Allow-KevFLIX-LAN.ps1 adds a narrowly scoped rule). No router changes are made.

## TV installation

Version 0.4.46 remembers a successful PC address and opens directly into KevFLIX on future launches. The address form appears on first use, when the PC is unreachable, or from Settings > Change PC address. A paired TV keeps its pairing and starts on Home. Update the TizenBrew module or standalone WGT to receive bootstrap/launcher changes; worker-only interface updates do not replace the TV launcher.

Target API baseline: Tizen 6.5 or newer. TizenBrew integration was written against its documented application-module format and the v2.0.5 release; installation has not been exercised.

- TizenBrew GitHub module: add malov-dot/kevflix. This launch route fetches public module assets over the internet.
- Offline-capable launcher: install the unsigned WGT from the latest GitHub release after signing/resigning through the supported TV installation workflow. TizenBrew Installer supports local WGT installation workflows; certificate requirements depend on the TV and installation method.

The standalone launcher subsequently loads its player interface from the local PC. It does not fetch media, translation or speech generation from a cloud service. It requires the PC worker to be running.

## Playback

TV Settings has Connect with: Plex or Local movie folders. Libraries from the selected local Plex server appear directly on the left. Choose a library, filter by genre/year/collection, sort, search or use the right-hand letter index when sorted A-Z. This reads Plex metadata; it does not copy the Plex client's pinned ordering or online streaming services. Non-video libraries can be browsed; video playback is the current scope.

Poster tiles show titles only underneath with watched/progress overlays. Movie details include synopsis, year, runtime, ratings, genres, cast/crew and media formats. Resume and Play from beginning are on that page. Per-movie PC processing controls are available before and during playback. Embedded text subtitle tracks are selected after opening playback; image subtitle formats are not currently rendered.

Browse Movie folders, or choose Plex library after configuring Plex access and path mappings. KevFLIX asks the PC to encode the source; it does not request Plex transcoding. Enable English dubbing in the player. The picture and English soundtrack are muxed together on the PC, and playback waits for complete segments.

Settings include output limit, Lanczos upscaling, HDR handling, audio tracks, text subtitles, commentary mode and optional Plex progress synchronization. Plex server administration remains in Plex's own local web interface. The optional progress setting uses Plex's normal API; no direct database edits occur.

## Sem

The sem/ directory contains a candidate generator and optional integration. The example config is generic. Stage against the current Sem source, compare continuity requirements, and obtain the required owner acceptance before activation. No Sem runtime is included here.

## Removal

Stop KevFLIX with Stop-KevFLIX.cmd and remove the TV launcher/module. Its cache and pairing records are in your KevFLIX profile. Original media and the Plex installation remain separate. The discarded transcoder-wrapper adapter is not part of this release and must not be installed.
