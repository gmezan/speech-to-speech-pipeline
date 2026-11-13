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
`facebook/mms-tts-quz`,
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
row's own `(text_quz, text_spa)` content, **not** a random
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

`--limit` caps eval rows per direction for a quick smoke test. Pass
`--revision` to pin a specific tagged commit of `--model` (e.g.
`--revision epoch-2`) rather than whatever's currently on `main` — useful
when several training runs have pushed to the same Hub repo over time (see
`nlp/translation/train.py`'s `--tag`) and you want to score one of them by
name even after later runs moved `main` past it. Each run
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
- NLLB's `quz_Latn` code is Cusco-Collao Quechua, an exact match for this
  project's target variety (unlike the earlier Chanka setup, which only had
  an approximate code to work with) — so this dialect-mismatch caveat no
  longer applies, as long as the eval data itself is genuine `quz`.

## `eval_golden.py` — medical-domain out-of-domain check

```sh
uv run eval_golden.py --model gmezan/nllb-quechua-spanish --revision epoch-3
```

Same BLEU/chrF++ scoring as `eval_mt.py`, but against
[`LosQuechuas/quechua-spanish-parallel-golden`](https://huggingface.co/datasets/LosQuechuas/quechua-spanish-parallel-golden)
— a small (61-row), hand-curated, medical-domain golden set, used purely as
an **out-of-domain generalization check**, not for training (too few rows
to fine-tune on safely). All training/eval data so far
(`americasnlp2024/jw300`) is religious-register text, so a steep quality
drop here relative to the main `test`-split scores is expected, not a bug
— this just measures how steep.

Note: this external dataset predates the project's `quy`→`quz` rename and
still uses the old `text_quy` column name — the *content* is genuine
Collao Quechua, just under the old header (`eval_golden.py` hardcodes this
override, nothing to configure). It also has no native splits (one `train`
split, no `test`) and no `source` column, both handled automatically.

## `compare_models_golden.py` — compare several models on the golden set in one pass

```sh
uv run compare_models_golden.py
```

Runs `eval_golden.py`'s scoring across several `label=repo_id[@revision]`
entries (default: the untuned base model plus `epoch-1`/`epoch-2`/`epoch-3`
of `$HF_USERNAME/nllb-quechua-spanish`) and writes one comparison table
(`golden_summary.md` by default) instead of eyeballing separate runs. Each
model's full results still append to `--results-path`
(`golden_results.jsonl` by default). Override the model list with
`--models`:

```sh
uv run compare_models_golden.py --models "mine=gmezan/nllb-quechua-spanish@epoch-2" "other=some/other-model"
```

## `publish_model_card.py` — document eval results on the model card

```sh
uv run publish_model_card.py --model gmezan/nllb-quechua-spanish --label epoch-2 --results-path mt_results.jsonl
```

Takes the last line of a `eval_mt.py` results JSONL and writes its full
per-direction/per-source BLEU/chrF++ table onto `--model`'s Hub model card
(`README.md`), under a block scoped to `--label` (needs write access, i.e.
`HF_TOKEN`). Re-running with the same `--label` replaces that block in
place rather than duplicating it; different labels accumulate side by side
under one `## Evaluation results` section — so a model repo that's been
pushed to multiple times (e.g. once per epoch, tagged via
`nlp/translation/train.py`'s `--tag`) ends up with every epoch's full eval
breakdown documented on the same card, not just whatever was pushed last.
`--label` is just a section name and doesn't have to match a real Hub
revision, but normally should (e.g. `epoch-2`) so the card's claims are
easy to verify by re-running `eval_mt.py --revision epoch-2`.

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
