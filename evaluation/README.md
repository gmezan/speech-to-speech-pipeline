# evaluation

Automatic evaluation for the models `stt/`, `tts/`, and `nlp/translation/`
produce, pulling everything straight from the Hugging Face Hub (same
`.env`/`HF_TOKEN`/`HF_USERNAME` pattern as the other modules).

## Setup

```sh
uv sync
```

## `eval_stt.py` — speech-to-text

```sh
uv run eval_stt.py
```

Loads the raw parallel dataset's native `test` split (default
`$HF_USERNAME/quechua-spanish-parallel`, see
`preprocessing/scripts/assign_splits.py`) — a fixed benchmark `stt/main.py`
never trains or validates against — transcribes it with `--model` (default
`$HF_USERNAME/whisper-quechua`), and reports **WER** and **CER** against the
reference transcripts.

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
**round-trip intelligibility** instead: take the text from the same native
`test` split (`tts/main.py` trains on `train`, validates on `validation`,
never touches `test`), synthesize it with `--tts-model` (default
`facebook/mms-tts-quy`,
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

## `eval_mt.py` — machine translation

```sh
uv run eval_mt.py
```

Loads the raw parallel dataset's native `test` split (default
`$HF_USERNAME/quechua-spanish-parallel`) — assigned once per row by
`preprocessing/scripts/assign_splits.py` as a deterministic hash of that
row's own `(text_quy, text_spa)` content, **not** a random
`train_test_split` recomputed at eval time. That means this benchmark set is
fixed regardless of how much the dataset grows later, and `translation/
train.py` only ever loads `train`/`validation`, so no training decision
(hyperparameters, checkpoint choice, early stopping off eval loss) can have
indirectly optimized against the rows scored here. Translates the test
split with `--model` (default `$HF_USERNAME/nllb-quechua-spanish`, pass
`facebook/nllb-200-distilled-600M` for the untuned base model, or a local
path for a pruned checkpoint), and reports **BLEU** and **chrF++** per
direction, broken down by the dataset's `source` column (plus an `ALL` row
per direction) so you can see whether quality differs across the mix of
underlying corpora (e.g. `americasnlp2024/jw300` vs `americasnlp2024/minedu`)
rather than one blended number hiding a domain skew.

Both hypothesis and reference are normalized before scoring (`“ ” ‘ ’ — –` →
plain `" ' - -`, zero-width characters stripped) — these six characters
decode to NLLB's `<unk>` token (verified against the actual tokenizer, not
byte-fallback, outright information loss) and are common in
`americasnlp2024/jw300`, so without normalizing, formatting differences
between the corpus and a model's output would get counted as translation
errors rather than the data-encoding issue they actually are.

```sh
uv run eval_mt.py --model facebook/nllb-200-distilled-600M --limit 50
```

`--limit` caps eval rows per direction for a quick smoke test. Each run
appends a JSON line (model, scores per direction/source, sentences/sec) to
`mt_results.jsonl` (configurable via `--results-path`, gitignored — it's a
local run log, not a committed artifact), so repeated runs across the base,
fine-tuned, and pruned checkpoints build up a comparable history instead of
one-off numbers. Run `uv run eval_mt.py --help` for the full list of options.

**Caveats:**
- If the dataset doesn't have native train/validation/test splits yet (or
  you're pointing at a different dataset), run
  `preprocessing/scripts/assign_splits.py` first — see its `--help` for the
  80/10/10 stratified-by-source assignment it does.
- As of now `americasnlp2024/jw300` is ~99% of the dataset, so the `ALL`
  row is almost entirely JW300's register (religious text); the
  `americasnlp2024/minedu` and any other smaller `source` breakdown is
  the more informative signal for general-domain quality, despite its
  much smaller sample size.
- NLLB's `quy_Latn` code is Ayacucho Quechua; this project otherwise
  targets the Chanka variant. A dialect mismatch between the eval data and
  the model's actual target variant would cap every model's score
  regardless of fine-tuning quality — worth a manual spot-check before
  trusting small deltas between runs.

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
