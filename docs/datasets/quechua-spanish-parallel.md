# `gmezan/quechua-spanish-parallel`

A parallel Quechua (Chanka, `quy`) ↔ Spanish (`spa`) corpus published on the
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
| `id` | `string` | `quy-NNNNNN`, sequential, unique, stable across pushes |
| `text_quy` | `string` | Quechua (Chanka) text |
| `audio_quy` | `Audio(sampling_rate=16000)` | Quechua speech recording of `text_quy`, where available |
| `text_spa` | `string` | Spanish text — same meaning as `text_quy` |
| `audio_spa` | `Audio(sampling_rate=16000)` | Spanish speech recording of `text_spa`, where available |
| `speaker_quy` / `speaker_spa` *(optional)* | `string` | recorder/speaker id, where known |
| `split` *(optional)* | `string` | `"train"` / `"validation"` / `"test"`, where fixed at the data level |
| `source` | `string` | which upstream dataset this row came from — see below |

## Sources merged so far

| `source` value | rows | has audio | notes |
|---|---:|---|---|
| `josemercado/quechua-chanka-asr` | 500 | `audio_quy` only | Original Quechua ASR corpus. Bootstrapped via `preprocessing/scripts/bootstrap_init_dataset.py`. `text_spa`/`audio_spa` are blank — Spanish side not yet collected for these rows. |
| `americasnlp2024/jw300` | 125,008 | none (text-only) | JW300 religious-text parallel sentences from the [AmericasNLP 2024 shared task](https://github.com/AmericasNLP/americasnlp2024), path `ST1_MachineTranslation/data/quechua-spanish/parallel_data/es-quy/jw300.es-quy.{es,quy}`. |
| `americasnlp2024/minedu` | 643 | none (text-only) | Peru Ministry of Education (MINEDU) parallel sentences, same shared-task repo, `minedu.quy-es.{es,quy}`. |

Total: 126,151 rows as of this writing.

Not merged: the same AmericasNLP directory also has a `dict_misc.quy-es.*`
pair (short dictionary-style entries rather than full sentences) — left out
to keep this corpus sentence-level. Pull it in the same way described below
if it's ever needed.

## Loading the whole thing

```python
from datasets import load_dataset

ds = load_dataset("gmezan/quechua-spanish-parallel")["train"]
```

## Filtering to one source

Every row's provenance is in `source`, so slice by it directly:

```python
jw300_only = ds.filter(lambda r: r["source"] == "americasnlp2024/jw300")
asr_only   = ds.filter(lambda r: r["source"] == "josemercado/quechua-chanka-asr")
```

## Picking the right rows per use case

Not every source is useful for every task — check which columns are actually
populated before using a source for a given purpose:

- **Text-only MT training** (`text_quy` ↔ `text_spa`): use all rows regardless
  of source; `americasnlp2024/jw300` and `americasnlp2024/minedu` are
  text-only to begin with, and `josemercado/quechua-chanka-asr` rows are
  currently unusable for MT since `text_spa` is still blank there.
- **ASR / TTS** (needs `audio_quy` or `audio_spa`): only
  `josemercado/quechua-chanka-asr` rows have audio today (`audio_quy` only,
  `audio_spa` still empty for all rows) — filter to that `source` first.
- **Domain-aware splits**: `jw300` is large and skews religious/formal
  register; `minedu` is small and skews educational. If you care about
  register balance (e.g. don't want JW300's ~125k rows to dominate training),
  stratify or subsample by `source` rather than relying on a random split.

## Adding another source later

1. Confirm the new source's rows are meaning-aligned Quechua↔Spanish pairs
   (or document which language is actually populated, as with the ASR rows).
2. Map its columns onto this schema, leaving unavailable fields (`audio_*`,
   `speaker_*`, `split`) as `None`/empty rather than omitting them.
3. Set a `source` string that identifies the origin (we use
   `owner/dataset-name` or `shared-task/corpus-name` style strings — keep that
   convention so filtering by `source` stays predictable).
4. Continue the `id` sequence (`quy-NNNNNN`) from the current row count rather
   than restarting it, and `concatenate_datasets` + `push_to_hub` onto the
   existing `train` split.
