# Training-ready output contract

`preprocessing/main.py` can point at any raw source dataset (`--dataset`,
`--audio-column`, `--text-column`), but whatever it points at, the thing it
writes to `--output-dir` must keep this exact shape. This is the contract
between preprocessing and every training script that consumes it (today:
`stt_tts/train_lora.py`; tomorrow: other ASR/TTS fine-tunes). Training code
should never need to change when the upstream dataset changes — only
preprocessing's CLI flags should.

## Format

A Hugging Face `datasets.Dataset`, always saved locally with
`Dataset.save_to_disk(path)` — an Arrow directory (`data-*.arrow`,
`dataset_info.json`, `state.json`), loaded back with
`datasets.load_from_disk(path)`.

Optionally also pushed to the Hub with `Dataset.push_to_hub(repo_id)`
(`preprocessing/main.py --push-to-hub --hub-repo-id ...`), so a training
step elsewhere just does `datasets.load_dataset(repo_id)` instead of needing
the local Arrow directory copied over. Same row schema either way — the Hub
repo is just parquet-backed storage for the same two columns, not a
different contract. See the [README](../README.md#pushing-to-the-hub) for
auth setup.

## Row schema

Exactly two columns, nothing else:

| column | type | shape | produced by |
|---|---|---|---|
| `input_features` | `float32` | `(80, 3000)` | `WhisperProcessor.feature_extractor` — log-Mel spectrogram, 80 mel bins over a fixed 30s window at 16kHz (padded/truncated inside the extractor, so this shape is constant regardless of source clip length) |
| `labels` | `int64`, variable length | `(seq_len,)` | `WhisperProcessor.tokenizer` — token ids for the transcript, unpadded (the training-time data collator pads dynamically and masks padding with `-100`) |

All other source columns (ids, raw audio, speaker, metadata, language) are
dropped at this stage (`remove_columns=dataset.column_names`). If you need to
carry metadata through to training/eval later, add it as an extra passthrough
column rather than changing the meaning of `input_features`/`labels` — a
consumer should be able to rely on those two keys always meaning what they
mean today.

## Invariants

- Audio is always resampled to 16000Hz mono before feature extraction,
  regardless of the source dataset's native sample rate.
- The `WhisperProcessor` used to build `input_features`/`labels` must be
  `from_pretrained`'d from the same checkpoint (and `language`/`task`) that
  will later fine-tune — a mismatched processor silently produces a different
  vocab and corrupts `labels`. The checkpoint is captured by convention in the
  output directory name, e.g. `quechua_whisper_ready` was built with
  `openai/whisper-small`.
- One input row produces exactly one output row (no merging/splitting), so
  row order has no particular meaning downstream — don't rely on positional
  alignment with the source dataset.

## One output directory per language/task, not one combined schema

This contract is single-language, single-task ASR (speech → tokens, anchored
to one Whisper decoder language). As the source corpus grows to cover more
than one language (see
[`dataset-schema.md`](./dataset-schema.md)), the right move is to run
`preprocessing/main.py` once per language — pointing `--audio-column`/
`--text-column` at `audio_quy`/`text_quy` or `audio_spa`/`text_spa` — and
write
each to its own output directory (e.g. `quechua_whisper_ready/`,
`spanish_whisper_ready/`). Don't fold a `language` column into this schema to
cover both at once: every consumer of this contract (the collator, the
trainer) assumes a single processor/vocab per dataset, and splitting at the
preprocessing stage keeps that assumption true without special-casing
training code.

A future text-to-text (translation) or audio-to-audio (S2S) training script
is a different consumer with a different contract — document its own
`input_ids`/`labels` or `input_features`/`target_features` shape next to its
training script rather than overloading this one.
