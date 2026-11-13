# speech-to-speech-pipeline

A Quechua (Collao variant) ↔ Spanish speech pipeline: collect aligned speech
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
`text_quz`/`audio_quz`, `text_spa`/`audio_spa`/`speaker_quz`/`speaker_spa`/
`split`) before either `stt/` or `tts/` can train — this repo's
[`gmezan/quechua-spanish-parallel`](https://huggingface.co/datasets/gmezan/quechua-spanish-parallel)
is populated this way, but currently text-only (both `text_quz` and
`text_spa` filled in, no audio yet on either side), so `stt/`/`tts/` can't
train against it until real `quz` audio is collected and merged in.

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

Fine-tunes Meta's `facebook/mms-tts-quz` (VITS) checkpoint on the same raw
Quechua corpus (`gmezan/quechua-spanish-parallel`, `text_quz`/`audio_quz`),
via a vendored copy of `ylacombe/finetune-hf-vits` (plain `transformers`
VITS has no training support). Verified end-to-end on a tiny smoke run
(`--limit 8`, CPU); a real training run (200 epochs, on a GPU box) and the
push to `gmezan/vits-quechua` have not been done yet — `app/` still falls
back to the untuned `facebook/mms-tts-quz` baseline.

### 5. [`app/`](./app) — playground

Gradio UI to manually test STT (`gmezan/whisper-quechua`) and TTS
(`facebook/mms-tts-quz`, baseline) **in isolation** — there is no speech ↔
speech wiring between them yet (see below). Both are Quechua-only, despite
`--language spanish` being a Whisper decoder-token anchor, not a sign of
Spanish training data — for real Spanish STT/TTS use the base
`openai/whisper-small`/`facebook/mms-tts-spa` instead (see
[`app/README.md`](./app/README.md#quechua-only--these-models-do-not-do-spanish)).

### 6. [`evaluation/`](./evaluation) — WER/CER, BLEU/chrF++

Automatic eval for every trained model, pulling the models/dataset from the
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
- **`eval_mt.py`** — scores a translation model's BLEU/chrF++ on the
  dataset's held-out `test` split, both directions:
  ```sh
  uv run eval_mt.py --model gmezan/nllb-quechua-spanish --revision epoch-3
  ```
- **`eval_golden.py`** / **`compare_models_golden.py`** — the same scoring
  against a small hand-curated medical-domain golden set
  ([`LosQuechuas/quechua-spanish-parallel-golden`](https://huggingface.co/datasets/LosQuechuas/quechua-spanish-parallel-golden)),
  used as an out-of-domain generalization check (all training data so far is
  JW300 religious-register text, so a quality drop here is expected — this
  measures how steep). `compare_models_golden.py` runs the base model plus
  every trained epoch in one pass and writes a single comparison table:
  ```sh
  uv run eval_golden.py --model gmezan/nllb-quechua-spanish --revision epoch-3
  uv run compare_models_golden.py
  ```
  See [`evaluation/README.md`](./evaluation/README.md) for the full option
  list and how both scripts handle that dataset's quirks (old `text_quy`
  column name, no native splits).

### 7. [`nlp/`](./nlp) — translation

[`translation/`](./nlp/translation) LoRA fine-tunes
`facebook/nllb-200-distilled-600M` — a single many-to-many model, not one
per direction — on `text_quz`/`text_spa` to translate both Quechua→Spanish
and Spanish→Quechua. Trained for 3 epochs so far (one per Kaggle session,
see [`infra/kaggle_train_epoch.ipynb`](./infra/kaggle_train_epoch.ipynb)),
each pushed and tagged (`epoch-1`/`epoch-2`/`epoch-3`) on
[`gmezan/nllb-quechua-spanish`](https://huggingface.co/gmezan/nllb-quechua-spanish) —
scores improve steadily (epoch-3: BLEU 7.96/13.28, chrF++ 33.09/31.59 on the
in-domain `test` split) with diminishing returns per epoch, consistent with
LoRA converging on this dataset size.

See [`docs/hardware-requirements.md`](./docs/hardware-requirements.md) for
disk/RAM/GPU sizing per model and variant, plus Quechua-only vs.
both-languages deployment totals.

### [`infra/`](./infra) — training compute

Shell notes for provisioning an ephemeral AWS EC2 GPU box, cloning the repo,
and running `stt/`'s and `tts/`'s `--push-to-hub` commands there for real
(non-smoke-test) training runs.

## What's done

- Aligned-recording tool (`speech_compilation/`) built and already used to
  produce the Quechua clips in the dataset.
- Parallel dataset schema designed and the Hub dataset
  (`gmezan/quechua-spanish-parallel`) now holds 136,589 genuine Quechua
  Collao (`quz`) ↔ Spanish text pairs from AmericasNLP2024's JW300 corpus
  (`text_quz`/`text_spa` both populated, no audio yet); every module now
  points at it.
- Whisper-ready preprocessing pipeline (`preprocessing/main.py`) built, run,
  and pushed (`gmezan/quechua-whisper-ready`).
- Whisper LoRA fine-tune (`stt/`) built, trained, merged, and pushed
  (`gmezan/whisper-quechua`).
- VITS fine-tuning toolkit (`tts/`) vendored, patched for current
  `transformers`, and verified correct on a smoke run.
- Standalone playground (`app/`) to manually exercise STT and TTS.
- WER/CER evaluation (`evaluation/`) for both the STT model directly and the
  TTS model via round-trip intelligibility; BLEU/chrF++ evaluation
  (`eval_mt.py`) for the translation model, including an out-of-domain
  medical check (`eval_golden.py`/`compare_models_golden.py`) against a
  separate hand-curated golden set.
- Translation (`nlp/translation/`) trained for real: LoRA fine-tune of
  `facebook/nllb-200-distilled-600M` for both Quechua↔Spanish directions in
  one model, 3 epochs run on Kaggle (one per session), each pushed and
  tagged on `gmezan/nllb-quechua-spanish`, with steadily improving
  in-domain BLEU/chrF++ each epoch.

## What's left

- **Collect real audio.** `audio_quz`/`audio_spa`/`speaker_*` are blank on
  every row — the current corpus is 136,589 genuine `quz`↔`spa` text pairs
  (JW300, AmericasNLP2024) with no recordings yet. `speech_compilation/` is
  built for recording aligned clips, but that pass hasn't happened for the
  `quz` variety. Until then, ASR (`stt/`) and TTS (`tts/`) have no `quz`
  training audio to use; only text-only translation (`nlp/translation/`) can
  train against this dataset today.
- **Run a real `tts/` training pass** on a GPU box (200 epochs, not the
  8-sample smoke test) and push the result to `gmezan/vits-quechua`, then
  point `evaluation/eval_tts.py` and `app/` at it instead of the
  `facebook/mms-tts-quz` baseline.
- **Wire `app/` into an actual speech-to-speech pipeline**: today STT and TTS
  are tested independently; there's no Quechua-audio-in →
  transcribe → (translate) → synthesize → Spanish/Quechua-audio-out flow.
- **Train translation further, or on more diverse data.** 3 epochs are
  trained and pushed, but all training/eval data so far is JW300
  (religious-register text) — per-epoch gains are already diminishing
  (LoRA converging on this data), and the model generalizes poorly to other
  domains (e.g. the medical golden-set check scores far below the in-domain
  `test` split). More epochs on the same data likely won't help much; more
  diverse `quz` data would.
- **No perceptual TTS quality score.** `evaluation/eval_tts.py` gives an
  automatic intelligibility proxy, not naturalness/MOS — that still needs
  human evaluation if it matters for this project.
- **No CI/CD.** `infra/` documents manual steps for provisioning a GPU box
  and kicking off training by hand; there's no automated pipeline (e.g. the
  "Pipeline 2" ephemeral-GPU-instance idea mentioned in `stt/README.md`) that
  triggers training when the dataset or code changes.
