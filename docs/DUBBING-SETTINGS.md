# Dubbing settings (2026-10-19 build)

Every key below is optional. A key that is absent keeps the earlier behaviour; `config.example.json` carries the values this build
was tuned and checked with. All of them live in `relay-config.json`.

## Rule that drives them

Every spoken line is dubbed. A line stays in the original language only when it is not speech (laughter, sound tags, music marks,
credit lines Whisper invents over silence, speech that is already English), when two people talk over each other, or when no
usable voice or recognizable words exist in its window. Subtitles are for viewing only and never reach the translator.

## `actor_beta` (per-line dubbing quality)

| key | tuned value | what it does |
|---|---|---|
| `subtitle_scoped` | `false` | dub whole windows from the audio, not only where a subtitle exists |
| `prefer_own_reference_seconds` | `3.0` | a line this long (or longer) is cloned from its own audio, so a whisper is dubbed quietly and a shout loudly |
| `vox_reference_only` | `true` | clone from the reference audio alone; never prompt Vox with the source-language transcript (it stretched English takes) |
| `bank_reference_min_seconds` | `2.0` | shortest clean reference accepted for a speaker's bank voice |
| `overlap_word_fraction`, `overlap_span_fraction` | `0.75`, `0.85` | how much of a word / line a second speaker must cover before the line counts as overlapped |
| `overlap_solo_relax` | `true` | a line flagged as overlapped when no second line was transcribed (laughter, noise) is dubbed |
| `unclear_min_probability` | `0.25` | mean recognizer confidence below which a line is unclear |
| `unclear_word_floor`, `unclear_low_word_share` | `0.2`, `0.5` | a word below the floor counts as low; a line is unclear when more than this share of its words are low |
| `merge_short_fragment_seconds` | `1.5` | a recognized fragment shorter than this ("Доча!") joins the same speaker's neighbouring fragment across a pause of up to 1 s (unit at most 6 s) so it can be cloned and read back |
| `reference_extend_merged_labels` | `true` | a line that is too short to be its own reference may borrow audio from its own continuous turn even when the diarizer's label looks merged |
| `borrow_reference` | `true` | a line that still cannot be its own reference borrows the admitted reference of another line of the same window (same diarization label first, then the nearest in time). The reviewed stable gate is unchanged; `relay/borrowed_reference.py` admits only declared borrows of a donor the gate itself admitted |
| `take_fit_max_speed` | `1.5` | the most a take is sped up (one pitch-preserving tempo change, hard ceiling 1.6) to fit its slot |
| `concise_retranslation` | `true` | when every take of a line is too long for its slot, the line is translated again with a word budget and rendered like any other take |
| `asr_local_gain` | on | quiet speech is raised for the recognizer only; the dub is mixed back at the original line's loudness |

| `language_correction` | `true` | the spoken language of every recognized segment is read from its audio; a segment whose audio language differs from the window's (>= 0.85) is decoded again in that language, and only when the new reading is itself confident (word confidence 0.70 or higher): two clean Spanish lines of the Spanish test film once became Turkish and English at 0.56-0.59. A window with an English scene otherwise makes Whisper write the Russian speech around it down as English words |
| `nonverbal_audio` | `true` | lines whose original audio is not speech to dub keep that audio: singing or chanted prayer, coughs, sneezes, sighs, screams without words, silence the recognizer invented text over (original level at or below -66 dB), and English spoken in a foreign film. Text rules (laughter, sound tags, music marks, credit lines, foreign scripts, vocalizations, English text) always apply; this key adds the acoustic rules (pitch, level, AudioSet labels). Calm spoken narration on one pitch is not a chant: a monotone line the classifier hears as Speech (0.60 or higher) at 2.5 words per second or faster is dubbed. The English-text rule does not count Spanish "has" or "come" as English |
| `deess_takes` | `false` (`true` in config.example.json) | Vox renders "s", "sh", "f" and "ch" as bursts that are on average 5 dB hotter than the actors' own and, in about one line in ten, louder than the vowels; played loudly they sound like static. With this on, each rendered take has its high band (above 5 kHz) lowered, only inside the frames where that band rises above 5 dB under the take's own voiced level, by at most 8 dB (relay/take_deess.py). Everything below 3.5 kHz and every take without a hot burst is returned unchanged; the result is a new take file (`-ds-` in its name) with its own hash, read back and mixed like any other take |
| `reject_unintelligible` | `true` | a take that still reads back as other words altogether after every seed (word error >= 1.0 and its letters resemble the intended words by less than 0.5) is not mixed in; the original line plays |

Top-level `audio_event_model`: folder of the AudioSet classifier `MIT/ast-finetuned-audioset-10-10-0.4593` (BSD-3-Clause, 346 MB: `config.json`,
`preprocessor_config.json`, `model.safetensors`). KevFLIX never downloads it; fetch it once with
`python -c "from huggingface_hub import snapshot_download as d; d('MIT/ast-finetuned-audioset-10-10-0.4593', local_dir=r'<folder>', allow_patterns=['config.json','preprocessor_config.json','model.safetensors'])"`.
Empty or absent: singing is then detected from pitch alone (fewer lines), everything else still applies.

Per line the engine also: renders further seeds (`7`, `1234`, `2024`) when a take overruns its slot, reads back wrongly or needs more than
1.3x tempo, and keeps the first take that fits at a natural pace and reads back clean; raises a take masked by loud music or effects
in 6 dB steps up to 12 dB instead of dropping it; asks the translator again with a word budget when it rambles, answers nothing or
answers with commentary instead of a translation; and chooses the voice a borrowing line uses by similar pitch first.

## Translation

`translation_engine` `qwen3`, `translation_llm_model` (Qwen3-8B folder, Apache-2.0), `translation_quantization` `int8`. The
default stays `m2m100` so an install without the model keeps working.

## Playback (`tv`)

`tv.loudness.enabled` and `tv.loudness.target_lufs` (`config.example.json` sets `-27`; the built-in default is the broadcast
`-23`): one constant gain per film, measured once on the original soundtrack. The film's own dynamics are kept; only the whole
film moves. Raise the target to make everything louder, lower it to make everything quieter.
`tv.actor_beta_start_seconds` (60) is the start buffer; the producer then keeps running to the end of the film and `mux_ahead_*`
keeps the segments ready, so nothing buffers during play.
