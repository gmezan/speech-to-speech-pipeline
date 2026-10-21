# evaluation

Automatic WER/CER evaluation for the models `stt/` and `tts/` produce, pulling
everything straight from the Hugging Face Hub (same `.env`/`HF_TOKEN`/
`HF_USERNAME` pattern as the other modules).

## Setup

```sh
uv sync
```

## `eval_stt.py` — speech-to-text

```sh
uv run eval_stt.py
```

Loads the raw parallel dataset (default `$HF_USERNAME/quechua-spanish-parallel`),
recovers the **same held-out eval split** `stt/main.py` trained against —
`dataset.train_test_split(test_size=--eval-size, seed=--seed)["test"]`, with
`--eval-size`/`--seed` defaulting to the same `0.1`/`42` `stt/main.py` uses —
transcribes it with `--model` (default `$HF_USERNAME/whisper-quechua`), and
reports **WER** and **CER** against the reference transcripts. This matters
because without it, you'd risk scoring the model on rows it already trained
on; `--eval-size`/`--seed` must be overridden together with whatever
`stt/main.py` was actually run with, if non-default.

```sh
uv run eval_stt.py --model openai/whisper-small --limit 20
```

`--limit` caps the number of eval rows for a quick smoke test. Run
`uv run eval_stt.py --help` for the full list of options.

## `eval_tts.py` — text-to-speech

```sh
uv run eval_tts.py
```

TTS has no reference audio to score against (it synthesizes new audio from
text, not a reproduction of an existing recording), so this measures
**round-trip intelligibility** instead: take the text from the same held-out
eval split, synthesize it with `--tts-model` (default `facebook/mms-tts-quy`,
the current baseline — swap to `$HF_USERNAME/vits-quechua` once `tts/`
pushes a fine-tuned one), transcribe the synthesized audio with
`--stt-model` (default `$HF_USERNAME/whisper-quechua`, the judge), and score
WER/CER between the synthesized-then-transcribed text and the original.

**Caveat:** this score is bounded by the judge STT model's own error rate —
a bad transcription of a *good* synthesis still looks bad. It measures
"can an ASR model understand what was synthesized," which is a reasonable
intelligibility proxy, but it is not a naturalness/prosody/MOS score. True
perceptual quality still needs human evaluation; that's out of scope here.

```sh
uv run eval_tts.py --tts-model gmezan/vits-quechua --limit 20
```

Run `uv run eval_tts.py --help` for the full list of options.

## Shared code

`common.py` has `pick_device()` (same `cuda` > `mps` > `cpu` logic as
`stt/`/`tts/`/`app/`) and `compute_error_rates()`, which normalizes text
(lowercase, strip punctuation/extra whitespace) before scoring so formatting
differences between the corpus and a model's output don't get counted as
errors.

## Auth: `.env` + `HF_USERNAME`

A `.env` file already exists in this directory (gitignored via the repo's
root `.gitignore`) with:

```
HF_USERNAME=<your-hf-username>
```

Add `HF_TOKEN=hf_...` too if any model/dataset you're evaluating is private.
