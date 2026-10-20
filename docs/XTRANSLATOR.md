# X-Translator + X-Voice movie dubbing

This engine accepts movie audio and produces English audio automatically. It
keeps X-Translator's session controller, speech detection, speaker changes,
recurring-speaker voice references, translation and X-Voice synthesis. Its
internal recognition and translation stages use text. Subtitles and manually
entered dialogue or speaker identities are not required.

KevFLIX supplies separated vocals and retains the background stem, maps output
to the movie clock, matches dialogue levels and buffers complete audio before
playback. Failed translations are reported as gaps with no dubbing credit.
Original foreign dialogue is not deliberately restored into those gaps.

## Installation

1. Install WSL2 with Ubuntu-24.04 and NVIDIA GPU support.
2. In PC settings, open **Install processing components** and install
   **X-Translator + X-Voice**. This downloads dependencies and model files into
   an isolated Linux runtime. It does not start inference or play a movie.
3. Reopen KevFLIX after installation so the saved runtime paths are loaded.
4. While playback and queued jobs are idle, select **X-Translator + X-Voice**
   as the dubbing engine. Choose automatic source-language detection, or a
   supported source-language setting. English is the output language.

For an explicit local profile, the same installer is available as:

```powershell
python -m relay.app_runtime.xt_provision --config C:\path\to\profile\config.json --distro Ubuntu-24.04
```

The installer preserves the current engine selection. Its model manifest and
runtime receipt are written under the profile's `xtranslator-install` folder;
`install.log` records progress. The Linux runtime defaults to `~/kevflix-xt`.
The ASR/MT environment is separate from X-Voice and from existing Sem runtimes.
The phonemizer is built into a private prefix without replacing system libraries.

## Runtime and playback

- Services bind to loopback and use local model assets after setup.
- One speaker-aware session persists across consecutive movie chunks.
- Source audio is fed once; context used for separation is not repeatedly fed
  to recognition.
- A completed processing barrier and terminal speech outcomes determine when
  a chunk may publish. Receiving audio alone does not mark it processed.
- Complete generated audio is retained across chunk boundaries. A worker
  restart is recorded as a speaker-state reset; already published audio remains
  immutable, while unpublished output from the old session is rejected.
- Model processes are owned by the movie GPU lease. Linux process cleanup is
  confirmed separately from Windows launcher exit. The runtime yields if Sem
  starts and preserves the configured free-memory floor, at least 3072 MiB.
- English source audio does not need translation. A silent opening leaves
  language detection pending until speech becomes available.

## Source and model provenance

- X-Translator: `zhaoyx239/X-Translator`, commit
  `dc6a59a867bd20fc9bb7aa3b360256af8d89cc5c`.
- X-Voice: `sunnyxrxrx/X-Voice`, commit
  `817bcdd9f8173eb01c7d5c895858b1d93780d9d0`.
- The vendored provenance records distinguish upstream source and local patches.
- Qwen3-ASR and LMT are separately provisioned recognition and translation models.
- X-Voice source is MIT; its pretrained weights are CC-BY-NC-4.0. Component
  model terms remain independent. Weights are not embedded in the app package.

## Evidence status

Version 0.4.35 is an implementation candidate. Source review and dependency
installation are not evidence of successful movie playback. Local throughput,
translation coverage, speaker consistency and listening quality require a
separate end-to-end run. Speech-time coverage is not word accuracy.

## 0.4.35 processing components

Movie audio drives the automatic ASR, internal text translation, and cloned
speech pipeline. Subtitles are for the viewer only and are excluded from model
input, timing decisions, and evaluation. The 0.4.35 candidate keeps sentence
units together and records why each unit was committed. A bounded soft language
prior comes only from qualifying native audio-recognition finals; it never
forces the returned language. If a queued voice reference was not yet available,
the synthesis stage refreshes that same speaker's reference once. Missing
references remain explicit failures instead of borrowing another voice.

Speaker-boundary and final-stop flushes retain their pending wake-up after the
last queued audio bytes are consumed. WebSocket failures retain peer close codes
and reasons for diagnosis. These changes do not alter recognition thresholds,
translation text, speaker decisions, or the demo's playback deadlines.

Owned Linux model processes allow a 4 GiB CUDA compilation cache, retaining the
existing persistent driver-cache location. This avoids forcing all kernels into
the nearly full default 1 GiB cache. It does not allocate VRAM, change models or
delete prior cache entries. Startup speed improvement requires measured evidence.

The movie input adapter conditions separated vocals for the upstream speech
thresholds using one bounded gain per window. This affects recognition and voice
reference input only, not the source stems or background mix. Stem receipts record
the gain, active level before and after conditioning, and limited sample count.
This is an input correction; improved translation coverage still requires a real
playback run and listening evaluation. Published chunk carry reads the immutable
receipt and database row identity, since the nested payload omits the job ID.
Generated speech uses the same active-frame RMS estimator on the output and
the original unboosted speech reference, avoiding whole-span silence bias.
Output gain remains capped at 2x, with the English-only headroom limiter and
background at unity. Input gain is not applied or inversely applied to playback.

Demucs separates dialogue from the background. Qwen3-ASR supplies recognition
and native language detection. The translation model is LMT-60-8B Q8_0 GGUF,
using the original LMT tokenizer and chat template with FP16 computation in
vLLM. X-Voice supplies the cloned English speech. This route does not construct
Whisper, Vox, or the legacy AnalysisService. Quantization applies to LMT;
X-Voice and Qwen3-ASR retain their existing model formats.

The Q8 conversion comes from mradermacher/LMT-60-8B-GGUF, revision
5db5f369e614a63c90c4b3100585ef15a0fe53d0. The installer verifies the pinned
file digest. Startup records estimated allocations and measured Demucs growth;
Q8 playback speed and translation quality remain unverified until a real run.

Speech-time accounting uses the upstream energy detector parameters on the
separated vocals. Its denominator differs from earlier Silero-based runs.
