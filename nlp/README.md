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
| `quz` | `quz_Latn` (Cusco-Collao Quechua) |
| `spa` | `spa_Latn` |

NLLB-200 ships with `quz_Latn` pretrained, so the base model has *some*
Quechua exposure already (fine-tuning specializes/improves it rather than
starting from zero). Unlike this project's earlier Chanka setup — where the
closest available NLLB code (`quy_Latn`, Ayacucho Quechua) was only an
approximation of the actual dialect — `quz_Latn` is an exact match for the
Collao variety this project now targets, so the fine-tune only has to close
a domain gap, not a dialect gap.

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
