# `gmezan/quechua-spanish-parallel`

A parallel Quechua (Collao, `quz`) ↔ Spanish (`spa`) corpus published on the
Hugging Face Hub at
[`gmezan/quechua-spanish-parallel`](https://huggingface.co/datasets/gmezan/quechua-spanish-parallel).
It aggregates rows from multiple source datasets into one schema, tagged by
origin via the `source` column, so it can grow over time without breaking
anything built on top of it.

See [`preprocessing/docs/dataset-schema.md`](../../preprocessing/docs/dataset-schema.md)
for the full column reference, the alignment contract, and how
`preprocessing/main.py` consumes this dataset. This doc covers provenance and
how to pull/aggregate specific sources.

## Columns

| column | type | notes |
|---|---|---|
| `id` | `string` | `quz-NNNNNN`, sequential, unique, stable across pushes |
| `text_quz` | `string` | Quechua (Collao) text |
| `audio_quz` | `Audio(sampling_rate=16000)` | Quechua speech recording of `text_quz`, where available |
| `text_spa` | `string` | Spanish text — same meaning as `text_quz` |
| `audio_spa` | `Audio(sampling_rate=16000)` | Spanish speech recording of `text_spa`, where available |
| `speaker_quz` / `speaker_spa` *(optional)* | `string` | recorder/speaker id, where known |
| `split` *(optional)* | `string` | `"train"` / `"validation"` / `"test"`, where fixed at the data level |
| `source` | `string` | which upstream dataset this row came from — see below |

## Sources merged so far

**Retired (Chanka-era, replaced):** the previous version of this dataset was
bootstrapped from Chanka/Ayacucho-variety sources — `josemercado/quechua-chanka-asr`
(500 audio rows) and the AmericasNLP2024 `quy` JW300/MINEDU files
(`es-quy/jw300.es-quy.{es,quy}`, `minedu.quy-es.{es,quy}`) — before the
project moved to Quechua Collao (`quz`). None of that is in the current
dataset.

| `source` value | rows | has audio | notes |
|---|---:|---|---|
| `americasnlp2024/jw300` | 136,589 | none (text-only) | JW300 religious-text parallel sentences, genuine Quechua Collao (`quz`) — from the [AmericasNLP 2024 shared task](https://github.com/AmericasNLP/americasnlp2024), path `ST1_MachineTranslation/data/quechua-spanish/parallel_data/es-quz/jw300.es-quz.{es,quz}`. Ingested via `preprocessing/scripts/bootstrap_jw300_quz.py`. |

Total: 136,589 rows as of this writing. No audio yet for any row — `audio_quz`/`audio_spa`
are blank until real `quz` ASR/TTS recordings are collected.

Note: AmericasNLP2024 only has a `quz` JW300 file — there's no `quz`
equivalent of the `minedu`/`dict_misc` sources (those exist only in `quy`),
so they aren't available to merge in the same way the old Chanka corpus did.

## Loading the whole thing

```python
from datasets import load_dataset

ds = load_dataset("gmezan/quechua-spanish-parallel")["train"]
```

## Filtering to one source

Every row's provenance is in `source`, so slice by it directly:

```python
jw300_only = ds.filter(lambda r: r["source"] == "americasnlp2024/jw300")
```

## Picking the right rows per use case

- **Text-only MT training** (`text_quz` ↔ `text_spa`): use all rows — the
  only source today (`americasnlp2024/jw300`) is text-only and fully
  populated on both sides.
- **ASR / TTS** (needs `audio_quz` or `audio_spa`): no source has audio yet;
  this needs a new `quz` recording/collection source added (see "Adding
  another source later" below) before ASR/TTS can use this dataset.
- **Domain skew**: all current rows are JW300 (religious/formal register) —
  worth keeping in mind until a more varied `quz` source is added.

## Adding another source later

1. Confirm the new source's rows are meaning-aligned Quechua↔Spanish pairs
   (or document which language is actually populated, as with the ASR rows).
2. Map its columns onto this schema, leaving unavailable fields (`audio_*`,
   `speaker_*`, `split`) as `None`/empty rather than omitting them.
3. Set a `source` string that identifies the origin (we use
   `owner/dataset-name` or `shared-task/corpus-name` style strings — keep that
   convention so filtering by `source` stays predictable).
4. Continue the `id` sequence (`quz-NNNNNN`) from the current row count rather
   than restarting it, and `concatenate_datasets` + `push_to_hub` onto the
   existing `train` split.
