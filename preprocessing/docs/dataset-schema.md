# Recommended schema: parallel Quechua–Spanish corpus

The live dataset, [`gmezan/quechua-spanish-parallel`](https://huggingface.co/datasets/gmezan/quechua-spanish-parallel),
follows this shape. It's bootstrapped (`scripts/bootstrap_jw300_quz.py`) from
genuine Quechua Collao (`quz`) text-only parallel sentences from the
AmericasNLP 2024 shared task's `jw300` source, which populate both
`text_quz` and `text_spa` but no audio yet. (An earlier version of this
dataset was bootstrapped from the Chanka-variety `josemercado/quechua-chanka-asr`
ASR corpus via `scripts/bootstrap_init_dataset.py`; that source has been
retired now that the project targets `quz`.) Each row's origin is recorded
in the `source` column — see
[`docs/datasets/quechua-spanish-parallel.md`](../../docs/datasets/quechua-spanish-parallel.md)
for the full provenance table and how to filter/aggregate by source. This
shape lets the raw data keep growing/changing without breaking
`preprocessing/` or anything built on top of it.

## Columns

| column | type | notes |
|---|---|---|
| `id` | `string` | stable unique id per aligned pair; use it to join/version across dataset revisions and to dedupe train/eval splits |
| `text_quz` | `string` | Quechua (Collao) transcript |
| `audio_quz` | `Audio(sampling_rate=16000)` | Quechua speech recording of `text_quz` |
| `text_spa` | `string` | Spanish transcript — same meaning as `text_quz`, not an independent sentence |
| `audio_spa` | `Audio(sampling_rate=16000)` | Spanish speech recording of `text_spa` |
| `speaker_quz` *(optional)* | `string` | recorder/speaker id for the Quechua take — enables speaker-aware TTS later and lets you check for speaker leakage across splits |
| `speaker_spa` *(optional)* | `string` | same, for the Spanish take |
| `source` | `string` | which upstream dataset this row came from (e.g. `josemercado/quechua-chanka-asr`, `americasnlp2024/jw300`); use it to filter to rows with the column availability a given use case needs |

Lowercase `text_<lang>` / `audio_<lang>` naming uses ISO 639-3 codes
(`quz` = Collao Quechua, `spa` = Spanish) — the same codes `speech_compilation`
already uses in its `LANGUAGE_NAMES` map, so column names and that app's
language codes stay consistent. It lets the same preprocessing run target
either language by just pointing `--text-column text_{lang} --audio-column
audio_{lang}` at the right pair, with no per-language branching in code. (The
current dataset's `TEXTO`/`audio` naming is a one-off quirk of the source
repo, not something to match.)

## Alignment, not concatenation

Each row is one aligned (Quechua, Spanish) utterance pair with equivalent
meaning — not two unrelated monolingual sentences sharing a row. That's what
makes the same corpus reusable for:

- **ASR** (either language): use that language's `audio_*`/`text_*` pair,
  ignore the rest — this is exactly what `preprocessing/main.py` does today.
- **TTS** (either language): same pair, reversed direction.
- **Translation**: `text_quz` ↔ `text_spa`, audio columns ignored.
- **Speech-to-speech**: `audio_quz` ↔ `audio_spa`, text columns ignored (or
  used as supervision/eval).

If rows aren't actually meaning-aligned, none of the cross-column use cases
work — alignment is the one property worth validating hardest when new data
comes in.

## Storage

Same convention as the current HF dataset: audio bytes embedded in Parquet
via `datasets.Audio(sampling_rate=16000)` cast columns, pushed with
`push_to_hub`. Keeps single-file portability and the dataset-versioning
rollback story already called out in
`stt_tts/docs/prerliminar-pipelines.md`.

## Where this plugs in

- **Collection**: `speech_compilation/main.py` already collects aligned pairs
  per entry (`id`, `text1`/`text2`, `audio_1`/`audio_2` keyed by the
  flashcards' two language columns) — it's already the right shape. The only
  gap is renaming/publishing its output as `text_quz`/`audio_quz`/`text_spa`/
  `audio_spa` instead of the generic `text1`/`text2` before pushing to the
  Hub.
- **Preprocessing**: `preprocessing/main.py` consumes one language's pair at
  a time and emits the ASR [training-ready contract](./output-schema.md)
  unchanged — run it twice (once per language), e.g.:

  ```sh
  uv run main.py --dataset <parallel-repo> \
    --text-column text_quz --audio-column audio_quz \
    --output-dir ./quechua_whisper_ready

  uv run main.py --dataset <parallel-repo> \
    --text-column text_spa --audio-column audio_spa \
    --output-dir ./spanish_whisper_ready
  ```
- **Future MT/S2S prep**: a separate script would read all four columns per
  row directly; it has its own output contract to document once it exists,
  independent of the ASR one.
