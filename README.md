# speech-to-speech-pipeline

A Quechua (Chanka variant) ↔ Spanish speech pipeline: collect aligned speech
data, fine-tune speech-to-text and text-to-speech models on it, and serve
them. Each stage is its own `uv`-managed Python project; `HF_TOKEN`/
`HF_USERNAME` (via a per-module `.env`) is the shared thread — every stage
reads its input from, and writes its output to, Hub repos namespaced under
`$HF_USERNAME` (default `gmezan`), so no stage needs a local copy of
another's output.

## Pipeline

```
speech_compilation/  →  preprocessing/        →  stt/  ──┐
  (record aligned         (Whisper-ready prep,              ├──→  app/ (playground)
   Quechua/Spanish          off an existing              │
   flashcard audio)         parallel dataset)     tts/  ──┴──→  evaluation/ (WER/CER)
```

### 1. [`speech_compilation/`](./speech_compilation) — data collection

A Gradio app for recording bilingual speech from flashcard prompts: a speaker
sees a prompt in two languages, records audio for each, can edit text and
replay before saving. Input is a semicolon-delimited flashcard CSV
(`data/flashcards_*.csv`); output is per-entry `.wav` files plus a
`flashcards.csv` of text + audio paths, resumable across runs. This is the
collection mechanism for the Spanish side of the parallel corpus (see below).

### 2. [`preprocessing/`](./preprocessing) — Whisper prep

A training dataset first needs to exist on the Hub following the
[parallel schema](./preprocessing/docs/dataset-schema.md) (`id`,
`text_quy`/`audio_quy`, `text_spa`/`audio_spa`/`speaker_quy`/`speaker_spa`/
`split`) before either `stt/` or `tts/` can train — this repo's
[`gmezan/quechua-spanish-parallel`](https://huggingface.co/datasets/gmezan/quechua-spanish-parallel)
is populated this way (Quechua side filled in, Spanish side still blank
pending collection).

`main.py` pulls `gmezan/quechua-spanish-parallel`, resamples audio to
16kHz, extracts log-Mel features and tokenizes labels with a
`WhisperProcessor`, and saves/pushes a training-ready dataset
(`input_features`/`labels` only — see the
[output contract](./preprocessing/docs/output-schema.md)) —
[`gmezan/quechua-whisper-ready`](https://huggingface.co/datasets/gmezan/quechua-whisper-ready).
Already run once for the Quechua side.

### 3. [`stt/`](./stt) — speech-to-text

LoRA fine-tunes `openai/whisper-small` on `gmezan/quechua-whisper-ready`,
pulling straight from the Hub. Merges the adapter into the base model and
pushes the full merged weights to
[`gmezan/whisper-quechua`](https://huggingface.co/gmezan/whisper-quechua),
so downstream inference is a plain `from_pretrained(repo_id)` with no PEFT
dependency. Quick-run on this machine's `mps`; real training targets a CUDA
box (see `infra/`). Already trained and pushed.

### 4. [`tts/`](./tts) — text-to-speech

Fine-tunes Meta's `facebook/mms-tts-quy` (VITS) checkpoint on the same raw
Quechua corpus (`gmezan/quechua-spanish-parallel`, `text_quy`/`audio_quy`),
via a vendored copy of `ylacombe/finetune-hf-vits` (plain `transformers`
VITS has no training support). Verified end-to-end on a tiny smoke run
(`--limit 8`, CPU); a real training run (200 epochs, on a GPU box) and the
push to `gmezan/vits-quechua` have not been done yet — `app/` still falls
back to the untuned `facebook/mms-tts-quy` baseline.

### 5. [`app/`](./app) — playground

Gradio UI to manually test STT (`gmezan/whisper-quechua`) and TTS
(`facebook/mms-tts-quy`, baseline) **in isolation** — there is no speech ↔
speech wiring between them yet (see below).

### 6. [`evaluation/`](./evaluation) — WER/CER

Automatic eval for both trained models, pulling the models/dataset from the
Hub like everything else:

- **`eval_stt.py`** — recovers the same held-out split `stt/`'s training run
  reserved for eval (same `train_test_split` seed/fraction over the raw
  dataset), transcribes it, and reports WER/CER against the reference text.
- **`eval_tts.py`** — TTS has no reference audio to diff against, so this
  measures round-trip intelligibility instead: synthesize eval-split text,
  transcribe the synthesized audio with the STT model as a judge, and score
  WER/CER against the original text. Bounded by the judge model's own error
  rate, and not a naturalness/MOS score — real perceptual quality still
  needs human eval, out of scope here.

### [`infra/`](./infra) — training compute

Shell notes for provisioning an ephemeral AWS EC2 GPU box, cloning the repo,
and running `stt/`'s and `tts/`'s `--push-to-hub` commands there for real
(non-smoke-test) training runs.

## What's done

- Aligned-recording tool (`speech_compilation/`) built and already used to
  produce the Quechua clips in the dataset.
- Parallel dataset schema designed and the Hub dataset
  (`gmezan/quechua-spanish-parallel`) bootstrapped with the Quechua side
  populated (500 rows); every module now points at it.
- Whisper-ready preprocessing pipeline (`preprocessing/main.py`) built, run,
  and pushed (`gmezan/quechua-whisper-ready`).
- Whisper LoRA fine-tune (`stt/`) built, trained, merged, and pushed
  (`gmezan/whisper-quechua`).
- VITS fine-tuning toolkit (`tts/`) vendored, patched for current
  `transformers`, and verified correct on a smoke run.
- Standalone playground (`app/`) to manually exercise STT and TTS.
- WER/CER evaluation (`evaluation/`) for both the STT model directly and the
  TTS model via round-trip intelligibility.

## What's left

- **Collect the Spanish side of the corpus.** `text_spa`/`audio_spa`/
  `speaker_*`/`split` are still blank on every row — `speech_compilation/` is
  built for this but the recording pass itself hasn't happened. Until then,
  translation and speech-to-speech use cases described in
  [`dataset-schema.md`](./preprocessing/docs/dataset-schema.md) aren't
  possible, only Quechua ASR/TTS.
- **Run a real `tts/` training pass** on a GPU box (200 epochs, not the
  8-sample smoke test) and push the result to `gmezan/vits-quechua`, then
  point `evaluation/eval_tts.py` and `app/` at it instead of the
  `facebook/mms-tts-quy` baseline.
- **Wire `app/` into an actual speech-to-speech pipeline**: today STT and TTS
  are tested independently; there's no Quechua-audio-in →
  transcribe → (translate) → synthesize → Spanish/Quechua-audio-out flow.
- **Translation — undecided.** `nlp/` is scaffolded but empty pending a call
  on whether this pipeline needs a translation step at all, or stays
  Quechua-only end-to-end. If it turns out to be needed, the plan is an
  open-weight MT model fine-tuned on `text_quy`/`text_spa` (once the Spanish
  side above is collected) rather than training one from scratch.
- **No perceptual TTS quality score.** `evaluation/eval_tts.py` gives an
  automatic intelligibility proxy, not naturalness/MOS — that still needs
  human evaluation if it matters for this project.
- **No CI/CD.** `infra/` documents manual steps for provisioning a GPU box
  and kicking off training by hand; there's no automated pipeline (e.g. the
  "Pipeline 2" ephemeral-GPU-instance idea mentioned in `stt/README.md`) that
  triggers training when the dataset or code changes.
