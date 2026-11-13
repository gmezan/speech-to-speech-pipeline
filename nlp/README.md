# nlp

Translation submodule: [`translation/`](./translation) LoRA fine-tunes
`facebook/nllb-200-distilled-600M` for Quechua (Collao) ↔ Spanish, pulling
the parallel dataset straight from the Hub (same pattern as `stt/`/`tts/`).

**Status: unblocked.** The dataset now has `text_spa` populated for every row
(the `quz` corpus from `americasnlp2024/jw300` is fully bilingual) — this
module hasn't been run for real yet, but there's no longer a data gap
stopping it.

## One model, both directions

NLLB-200 is a single many-to-many model (like M2M100/mBART), not one model
per direction: every input is tagged with a source-language token and the
decoder is forced to start with a target-language token, out of a shared
vocabulary covering all 200 languages. So `translation/train.py` fine-tunes
**one** checkpoint on both directions at once — for every aligned
`(text_quz, text_spa)` row it generates two training examples (quz→spa and
spa→quz, tags swapped) — rather than training two separate models.

### Language codes

| this project's code | NLLB FLORES-200 code |
|---|---|
| `quz` | `quy_Latn` (Ayacucho/Chanka Quechua) |
| `spa` | `spa_Latn` |

**Correction:** an earlier version of this doc claimed NLLB-200 ships a
`quz_Latn` (Cusco-Collao) code that's an exact match for this project's
target variety. That was wrong — verified directly against
`facebook/nllb-200-distilled-600M`'s tokenizer,
`convert_tokens_to_ids("quz_Latn")` returns the **UNK token id**, not a
real language embedding; `"quz_Latn"` isn't a pretrained NLLB code at all.
`quy_Latn` (Ayacucho/Chanka Quechua) is the real, closest available
pretrained code, same as this project's earlier Chanka setup used — so
there *is* a dialect gap between it and the Collao variety this project
targets, on top of the domain gap. An earlier training run (tagged
`epoch-1`/`epoch-2`/`epoch-3`, since retrained) used the fictitious
`quz_Latn` code, meaning every Quechua-tagged example was actually tagged
with UNK instead of a real language marker — it still trained and improved
run over run, but without leveraging NLLB's pretrained Quechua knowledge at
all.

## `translation/train.py` — fine-tuning

```sh
uv run translation/train.py
```

Pulls `--dataset` (default `$HF_USERNAME/quechua-spanish-parallel`), trains
on `--train-split` (default `train`) and validates on `--validation-split`
(default `validation`) — native HF splits, filtered to rows whose `source`
starts with one of `--source-prefixes` (default `["americasnlp"]`, an
explicit allow-list — currently only `americasnlp*` sources are cleaned of
`<unk>`-inducing characters, see `preprocessing/scripts/clean_text.py` —
rather than "everything with non-blank `text_spa`", so a future source
doesn't get pulled into training just by having both languages populated),
building the bidirectional pair set from `--text-column-quz`/
`--text-column-spa` (defaults `text_quz`/`text_spa`) for each. These splits
are fixed at the data level (see `preprocessing/scripts/assign_splits.py`,
80/10/10 stratified by `source`), shared with `stt/` and `tts/`, not
re-derived here, so `test` rows are never loaded by this script —
`evaluation/eval_mt.py` scores those. LoRA-fine-tunes `--model` (default
`facebook/nllb-200-distilled-600M`) for `--max-steps` (default 500), and
saves the adapter to `<output-dir>-adapter`.

Pass `--num-train-epochs N` instead of `--max-steps` to train for N full
passes over `--train-split` regardless of effective batch size — useful
since multi-GPU setups (e.g. Kaggle's T4×2) silently double throughput via
`DataParallel`, which a hardcoded step count can't account for.

```sh
uv run translation/train.py --push-to-hub
```

Merges the LoRA adapter into the base model and pushes the merged weights
plus tokenizer to `--hub-repo-id` (default
`$HF_USERNAME/nllb-quechua-spanish`), so downstream inference is a plain
`from_pretrained(repo_id)` with no PEFT dependency — same contract as
`stt/`. Run `uv run translation/train.py --help` for the full list of
options (LoRA rank/alpha/dropout/target modules, batch size, learning
rate, max length, etc).

Pass `--tag epoch-1` (or any label) alongside `--push-to-hub` to tag the
commit just pushed — lets you keep training the *same* `--hub-repo-id`
across multiple runs (e.g. one epoch per Kaggle session, each continuing
from the previous one's `main`) while still being able to pull any earlier
run back by name later (`from_pretrained(repo_id, revision="epoch-1")`),
even after newer commits have moved `main` past it. See
`infra/kaggle_train_epoch.ipynb` for this exact workflow end to end.

## `translation/translate.py` — quick manual testing

```sh
uv run translation/translate.py "Imaynallan kashanki?" --src quz --tgt spa
```

Translates one sentence with `--model` (default
`$HF_USERNAME/nllb-quechua-spanish`); pass
`--model facebook/nllb-200-distilled-600M` to compare against the untuned
base model. `--src`/`--tgt` pick the direction (`quz`/`spa`); both ways
work out of the same model.

## Auth: `.env` + `HF_USERNAME`

A `.env` file already exists in this directory (gitignored via the repo's
root `.gitignore`) with:

```
HF_USERNAME=<your-hf-username>
```

Add `HF_TOKEN=hf_...` too once any dataset/model involved is private.

## Device: Mac today, CUDA for real training

Same as `stt/`/`tts/`: auto-detects `cuda` > `mps` > `cpu`, only enables
`fp16` on `cuda`. A real training run belongs on the GPU box from
`infra/`, not this Mac.
