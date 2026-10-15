# Third-party provenance

KevFLIX uses the following upstream components. Their names and licenses remain unchanged. Model weights, runtime binaries, private configurations and development evidence are not included.

| Component | Selected version/revision | Upstream / license reference |
|---|---|---|
| faster-whisper | 1.2.1 | [SYSTRAN, MIT](https://github.com/SYSTRAN/faster-whisper) |
| Whisper small conversion | 536b0662742c02347bc0e980a01041f333bce120 | [Model card, MIT](https://huggingface.co/Systran/faster-whisper-small) |
| CTranslate2 | 4.6.0 | [OpenNMT, MIT](https://github.com/OpenNMT/CTranslate2) |
| M2M100 418M | 55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636 | [Meta card, MIT](https://huggingface.co/facebook/m2m100_418M) |
| Transformers | 4.57.1 | [Hugging Face, Apache-2.0](https://github.com/huggingface/transformers) |
| pyannote.audio | 4.0.7 | [pyannote, MIT](https://github.com/pyannote/pyannote-audio) |
| Community-1 | 3533c8cf8e369892e6b79ff1bf80f7b0286a54ee | [CC-BY-4.0 and repository access terms](https://huggingface.co/pyannote/speaker-diarization-community-1) |
| Demucs | 4.1.0 | [Meta, MIT](https://github.com/facebookresearch/demucs) |
| HTDemucs assets | cbc8a9b1a87023b7fd74e7b3412e6321c0eab003 | [Model repository](https://huggingface.co/adefossez/HTDemucs); retain its card and upstream attribution |
| Piper runtime | 1.8.0 | [OHF-Voice/piper1-gpl, GPL-3.0](https://github.com/OHF-Voice/piper1-gpl) |
| Piper libritts_r medium | piper-voices c10ece1aade47bb51c153c893d14e5bf8e5b7117 | [Voice card](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/libritts_r/medium/MODEL_CARD); English, 904 speakers, LibriTTS-R dataset CC-BY-4.0, fine-tuned from lessac medium |
| Lingua | 2.1.1 | [pemistahl, Apache-2.0](https://github.com/pemistahl/lingua-py) |
| pysubs2 | 1.8.0 | [tkarabela, MIT](https://github.com/tkarabela/pysubs2) |
| PyTorch / torchaudio | 2.11.0, CUDA 12.8 distribution | [PyTorch licenses/setup](https://pytorch.org/) |
| FFmpeg | private Gyan 9.0.2 full shared build, excluded from source archive | [Gyan build GPL-3.0 and notices](https://www.gyan.dev/ffmpeg/builds/); [FFmpeg licensing](https://ffmpeg.org/legal.html) |
| Windows SAPI | optional installed component | Installed voices; not redistributed |
| Plex / Samsung Plex | separately installed user software | Proprietary software; no binaries included |

Piper's current runtime is GPL, unlike the earlier MIT implementation. This package does not claim an all-MIT dependency stack. Preserve runtime and individual model notices before publication.

