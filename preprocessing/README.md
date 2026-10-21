# preprocessing

Prepares the [gmezan/quechua-spanish-parallel](https://huggingface.co/datasets/gmezan/quechua-spanish-parallel)
dataset (Chanka Quechua speech paired with `text_quy` transcriptions, see
[dataset-schema.md](./docs/dataset-schema.md)) for Whisper fine-tuning:
resamples audio to 16kHz, extracts log-Mel features, tokenizes labels, and
saves the result to disk with `datasets.save_to_disk`.

## Usage

```sh
uv run main.py
```

Override any default via flags, e.g. to target a different checkpoint or output path:

```sh
uv run main.py --model openai/whisper-medium --output-dir ./out
```

Run `uv run main.py --help` for the full list of options.

## Pushing to the Hub

So a training step can just `load_dataset(repo_id)` instead of needing the
local output directory, push the processed dataset straight to the Hub:

```sh
uv run main.py --push-to-hub
```

`--hub-repo-id` defaults to `$HF_USERNAME/quechua-whisper-ready` (override
with `--hub-repo-id` directly if needed). This needs write access to that
repo. Create a `.env` file in this directory (already covered by the repo's
root `.gitignore`, so it won't get committed) with:

```
HF_TOKEN=hf_...
HF_USERNAME=<your-hf-username>
```

`main.py` loads it automatically via `python-dotenv` before touching the
Hub — no need to `export` either in your shell. Add `--private` to create
the repo as private.

## Docs

- [Training-ready output contract](./docs/output-schema.md) — the stable
  `input_features`/`labels` shape this script must always produce, whatever
  dataset feeds it.
- [Recommended parallel dataset schema](./docs/dataset-schema.md) — proposed
  columns for the in-progress Quechua+Spanish parallel corpus.
