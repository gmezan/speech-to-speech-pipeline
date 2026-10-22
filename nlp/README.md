# nlp

Translation submodule: [`translation/`](./translation) LoRA fine-tunes
`facebook/nllb-200-distilled-600M` for Quechua (Chanka) ↔ Spanish, pulling
the parallel dataset straight from the Hub (same pattern as `stt/`/`tts/`).

**Status: blocked.** Translation needs the Spanish side of
[`gmezan/quechua-spanish-parallel`](https://huggingface.co/datasets/gmezan/quechua-spanish-parallel)
(`text_spa`), which isn't collected yet (see the root README's "What's
left"). This module is wired up and ready to run the moment that data
exists — it hasn't been run for real.

## One model, both directions

NLLB-200 is a single many-to-many model (like M2M100/mBART), not one model
per direction: every input is tagged with a source-language token and the
decoder is forced to start with a target-language token, out of a shared
vocabulary covering all 200 languages. So `translation/train.py` fine-tunes
**one** checkpoint on both directions at once — for every aligned
`(text_quy, text_spa)` row it generates two training examples (quy→spa and
spa→quy, tags swapped) — rather than training two separate models.

### Language codes

| this project's code | NLLB FLORES-200 code |
|---|---|
| `quy` | `quy_Latn` (Ayacucho Quechua) |
| `spa` | `spa_Latn` |

NLLB-200 ships with `quy_Latn` pretrained, so the base model has *some*
Quechua exposure already (fine-tuning specializes/improves it rather than
starting from zero). That pretraining is on **Ayacucho** Quechua
specifically — a closely related but not identical variety to the
**Chanka** Quechua this project's audio/text actually is — so expect the
fine-tune to need to close some dialect gap, not just a domain gap.

## `translation/train.py` — fine-tuning

```sh
uv run translation/train.py
```

Pulls `--dataset` (default `$HF_USERNAME/quechua-spanish-parallel`),
builds the bidirectional pair set from `--text-column-quy`/
`--text-column-spa` (defaults `text_quy`/`text_spa`), holds out
`--eval-size` (default 10%) for eval, LoRA-fine-tunes `--model` (default
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
uv run translation/translate.py "Imaynallan kashanki?" --src quy --tgt spa
```

Translates one sentence with `--model` (default
`$HF_USERNAME/nllb-quechua-spanish`); pass
`--model facebook/nllb-200-distilled-600M` to compare against the untuned
base model. `--src`/`--tgt` pick the direction (`quy`/`spa`); both ways
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
