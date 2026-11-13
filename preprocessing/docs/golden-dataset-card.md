---
language:
- qu
- es
task_categories:
- automatic-speech-recognition
- text-to-speech
- translation
tags:
- quechua
- spanish
- speech
- parallel-corpus
pretty_name: Quechua-Spanish Parallel Golden Dataset
dataset_info:
  features:
  - name: id
    dtype: string
  - name: text_spa
    dtype: string
  - name: text_quy
    dtype: string
  - name: audio_spa
    dtype:
      audio:
        sampling_rate: 16000
  - name: audio_quy
    dtype:
      audio:
        sampling_rate: 16000
  splits:
  - name: train
    num_bytes: 8682752
    num_examples: 61
  download_size: 8686475
  dataset_size: 8682752
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# Quechua-Spanish Parallel Golden Dataset

A hand-aligned, hand-checked parallel corpus of spoken Quechua (`quy`) and
Spanish (`spa`) utterances. Each row is one aligned pair: a Quechua recording,
its Spanish translation recording, and both transcripts — so the same corpus
drives ASR, TTS, speech-to-speech, and text translation training without
reshaping.

This is the **golden** dataset: the curated ground truth other
training-ready, task-specific datasets (e.g. Whisper-ready ASR manifests) are
derived from downstream. It intentionally carries no `split` column — splits
are a training-time decision, not a property of the source data — and no
`source` column, since every row here comes from one single-origin batch of
recordings rather than an aggregation of multiple upstream sources.

## Columns

| column | type | notes |
|---|---|---|
| `id` | `string` | stable unique id per aligned pair |
| `text_spa` | `string` | Spanish transcript |
| `text_quy` | `string` | Quechua (Ayacucho/Chanka, ISO 639-3 `quy`) transcript — same meaning as `text_spa`, not an independent sentence |
| `audio_spa` | `Audio(sampling_rate=16000)` | Spanish speech recording of `text_spa`, resampled/processed to 16kHz mono |
| `audio_quy` | `Audio(sampling_rate=16000)` | Quechua speech recording of `text_quy`, resampled/processed to 16kHz mono |

No `speaker_spa`/`speaker_quy` columns: the source recordings for this batch
don't carry reliable per-speaker identity, so no speaker id is published
rather than guessing one.

## Alignment, not concatenation

Each row is one aligned (Quechua, Spanish) utterance pair with equivalent
meaning, not two unrelated monolingual sentences sharing a row:

- **ASR** (either language): use that language's `audio_*`/`text_*` pair.
- **TTS** (either language): same pair, reversed direction.
- **Translation**: `text_quy` ↔ `text_spa`, audio ignored.
- **Speech-to-speech**: `audio_quy` ↔ `audio_spa`, text ignored or used as
  supervision/eval.

## Loading

```python
from datasets import load_dataset

ds = load_dataset("LosQuechuas/quechua-spanish-parallel-golden")["train"]
```

## Provenance

Built from a preprocessing run's `manifiesto.csv` manifest and its paired
16kHz-processed audio directories (`audio_16k/{quy,spa}/<id>.wav`) via
[`preprocessing/scripts/build_golden_dataset.py`](https://github.com/gmezan/speech-to-speech-pipeline/blob/main/preprocessing/scripts/build_golden_dataset.py)
in the [`speech-to-speech-pipeline`](https://github.com/gmezan/speech-to-speech-pipeline)
repo. Only rows the manifest marked as included (`incluido == True`) are
present — pairs that failed alignment or audio-quality checks upstream are
excluded.

See `preprocessing/docs/dataset-schema.md` in that repo for the broader
schema conventions this dataset follows.
