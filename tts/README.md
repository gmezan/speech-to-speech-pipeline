# tts

Fine-tunes a VITS text-to-speech model on the Quechua speech corpus, using
[Coqui TTS](https://github.com/idiap/coqui-ai-TTS) (`coqui-tts` — the
maintained fork of the library already used for inference in
`tmp_stt_tts/app.py`/`app_trained.py`).

## Same corpus as `stt/`, different shape

This is the **same underlying Quechua speech corpus** `stt/` trains on
([`josemercado/quechua-chanka-asr`](https://huggingface.co/datasets/josemercado/quechua-chanka-asr)),
but pulled in its **raw** form — waveform + plain transcript — rather than
the Whisper-preprocessed form `preprocessing/` produces
(`gmezan/quechua-whisper-ready`, `input_features`/`labels`).

That preprocessed dataset can't be reused here: it strips the raw audio
entirely and tokenizes text against Whisper's vocabulary, which is specific
to ASR (see `preprocessing/docs/output-schema.md`'s "One output directory
per language/task" section). TTS needs the opposite direction — raw
waveform as the *target*, plain text as the *input* — so `tts/main.py`
loads the raw dataset directly instead of going through `preprocessing/`.

## Usage

```sh
uv run main.py
```

This pulls `--dataset` (default `josemercado/quechua-chanka-asr`, split
`train`), decodes and resamples every clip to `--sample-rate` (default
22050Hz, VITS' standard), and materializes it on disk as a LJSpeech-style
manifest under `--data-dir` (default `./tts_data/wavs/*.wav` +
`./tts_data/metadata.csv`) — Coqui's trainer reads from disk, not from an
in-memory HF `Dataset`. It then builds a VITS config, a *grapheme*-based
tokenizer sized from whatever characters actually appear in the transcripts
(Quechua/Spanish text — ñ, á, í, apostrophes for ejectives, etc. — not the
plain-English A–Z default), and trains with Coqui's `Trainer`.

Already materialized `--data-dir` from a previous run? Skip re-downloading
and re-decoding audio:

```sh
uv run main.py --skip-materialize --data-dir ./tts_data
```

For a quick correctness check instead of a real run, cap the dataset size:

```sh
uv run main.py --limit 16 --batch-size 2 --eval-batch-size 2 --epochs 1 \
  --print-step 1 --save-step 100000
```

Run `uv run main.py --help` for the full list of options (LoRA-equivalent
knobs here are VITS/trainer settings: batch size, epochs, save/print steps,
eval split size, `--restore-path` to continue from an existing checkpoint
instead of training from scratch).

## No phonemizer

`use_phonemes=False` by default — training is grapheme-based (raw
characters), because there's no reliable Quechua phonemizer backend for
`espeak-ng`. The character set is derived automatically from the actual
training transcripts each run, not hardcoded, so it adapts if the corpus
changes (e.g. once the parallel Spanish text lands, see
`preprocessing/docs/dataset-schema.md`).

## Pushing the trained model to the Hub

```sh
uv run main.py --push-to-hub --hub-repo-id <your-username>/vits-quechua
```

Uploads the run's checkpoints + `config.json` (via `huggingface_hub`,
skipping TensorBoard event logs). Add `--private` to create the repo as
private. Needs write access via `HF_TOKEN` (see below).

## Auth: `.env` + `HF_TOKEN`

A `.env` file already exists in this directory (gitignored via the repo's
root `.gitignore`) with:

```
HF_TOKEN=hf_...
```

`main.py` loads it automatically via `python-dotenv` at startup; both
`datasets.load_dataset` and the Hub upload pick up `HF_TOKEN` from the
environment with no explicit token argument needed.

## Device: Mac today, CUDA for real training

Coqui's `Trainer` only accelerates on CUDA or runs on CPU — unlike `stt/`'s
Whisper fine-tuning (via `transformers`), it has no Apple `mps` backend. That
means:

- **On this Mac**: training runs on CPU. Fine for the `--limit`-capped
  smoke test above to confirm the pipeline works end-to-end; far too slow
  for a real VITS run (GAN-style training over the full corpus).
- **On a Linux box with an NVIDIA GPU**: same command, same code — just run
  `uv sync` there and drop `--limit`. `--mixed-precision` defaults on
  automatically when CUDA is available. This is "Pipeline 2" from the
  project's CI/CD design: an ephemeral GPU instance pulls the raw dataset,
  runs this script, and pushes the resulting model back to the Hub.

## Verified

`uv sync`, `--help`, the `--push-to-hub` validation error path, and a full
tiny end-to-end run (`--limit 8`, 1 epoch, CPU) all succeed: materializes
wavs + metadata, builds the character set, trains, evaluates, and saves a
checkpoint.
