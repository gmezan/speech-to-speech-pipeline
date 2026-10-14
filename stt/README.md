# stt

Fine-tunes Whisper (LoRA) for speech-to-text, pulling its input directly from
the Hugging Face Hub instead of a local copy:

[`gmezan/quechua-whisper-ready`](https://huggingface.co/datasets/gmezan/quechua-whisper-ready)
(produced by [`preprocessing/`](../preprocessing)) → fine-tune → merged model
pushed back to the Hub.

This mirrors `preprocessing/`'s auth pattern: a local `.env` with `HF_TOKEN`,
loaded automatically, no shell `export` needed.

## Usage

```sh
uv run main.py
```

This pulls `--dataset` (default `gmezan/quechua-whisper-ready`, split
`train`), holds out `--eval-size` (default 10%) for eval, LoRA-fine-tunes
`--model` (default `openai/whisper-small`) for `--max-steps` (default 500),
and saves the adapter to `<output-dir>-adapter` (default
`./whisper-stt-lora-adapter`, derived from `--output-dir`, default
`./whisper-stt-lora`). Nothing is trained from or saved to a dataset file on
disk — only the Hub repo id matters.

Override anything via flags, e.g. for a longer run on a bigger box:

```sh
uv run main.py --max-steps 2000 --per-device-train-batch-size 16
```

Run `uv run main.py --help` for the full list (LoRA rank/alpha/dropout/target
modules, batch size, learning rate, warmup, logging/save/eval steps,
generation length).

## Pushing the trained model to the Hub

```sh
uv run main.py --push-to-hub --hub-repo-id <your-username>/whisper-quechua
```

This merges the LoRA adapter into the base model (`merge_and_unload`) and
pushes the full merged weights plus the processor — so downstream inference
(or `tts`'s pipeline) can just `from_pretrained(repo_id)` with no PEFT
dependency or separate adapter-loading step. Add `--private` to create the
repo as private. Needs write access to that repo via `HF_TOKEN` (see below).

## Auth: `.env` + `HF_TOKEN`

A `.env` file already exists in this directory (gitignored via the repo's
root `.gitignore`) with:

```
HF_TOKEN=hf_...
```

`main.py` loads it automatically via `python-dotenv` at startup, and both
`datasets.load_dataset` and `push_to_hub` pick up `HF_TOKEN` from the
environment with no explicit token argument needed.

## Device: Mac today, CUDA for real training

`main.py` auto-detects `cuda` > `mps` > `cpu` and only enables `fp16` on
`cuda` (mixed precision isn't reliable on `mps`, and unsupported on `cpu`).
That means:

- **On this Mac**: runs on `mps`, useful for a quick correctness check with
  a small `--max-steps` — not a real training run.
- **On a Linux box with an NVIDIA GPU**: same command, same code, just run
  `uv sync` there (installs a CUDA-capable `torch` build for that platform)
  and point `--max-steps`/batch size at whatever the GPU can handle. This is
  "Pipeline 2" from the project's CI/CD design — an ephemeral GPU instance
  pulls the dataset from the Hub, runs this script, and pushes the resulting
  model back.

## Output contract

Consumes exactly the two columns `preprocessing/` is documented to always
produce — `input_features` (log-Mel spectrogram) and `labels` (tokenized
transcript) — so it never needs to change when the upstream raw dataset
changes, only `preprocessing/`'s CLI flags do. See
[`../preprocessing/docs/output-schema.md`](../preprocessing/docs/output-schema.md).
