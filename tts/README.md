# tts

Fine-tunes [Meta's MMS-TTS](https://huggingface.co/facebook/mms-tts) VITS
checkpoint for Quechua — [`facebook/mms-tts-quy`](https://huggingface.co/facebook/mms-tts-quy),
the exact Ayacucho/Chanka Quechua variant our data is in — on the Quechua
speech corpus. Training uses a vendored copy of
[`ylacombe/finetune-hf-vits`](https://github.com/ylacombe/finetune-hf-vits)
(MIT licensed, see [`finetune_hf_vits/LICENSE`](./finetune_hf_vits/LICENSE)),
since plain `transformers` ships VITS for inference only — it has no
training/discriminator support (`VitsModel.forward` raises
`NotImplementedError` if you pass `labels`). `finetune_hf_vits/` reimplements
the VITS generator+discriminator GAN training loop around the same
`transformers` checkpoint format, using `accelerate`.

## Why not Coqui, why not LoRA

Earlier iterations of this module tried training VITS from scratch with
Coqui TTS, then considered LoRA. Neither fit:

- **From scratch** needs far more data/steps than our ~500-clip corpus can
  support, and ignores a pretrained checkpoint that already exists for this
  exact language.
- **LoRA** doesn't have clean target modules on VITS's
  convolutional/flow architecture the way it does on a transformer like
  Whisper, and no LoRA-ready Quechua VITS checkpoint exists to build on.

Fine-tuning the real `facebook/mms-tts-quy` checkpoint is the better fit:
purpose-built for this language, and the toolkit's own benchmark claims a
good result in "20 minutes with as little as 80 to 150 samples" — well within
what our corpus provides. Caveat: MMS checkpoints are
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (non-commercial)
— any finetuned model inherits that license.

## Same corpus as `stt/`, raw form

Like `stt/`, this trains on the same Quechua speech corpus
([`gmezan/quechua-spanish-parallel`](https://huggingface.co/datasets/gmezan/quechua-spanish-parallel),
`text_quy`/`audio_quy` columns — see
[dataset-schema.md](../preprocessing/docs/dataset-schema.md)),
pulled in its raw form (audio + transcript), not the Whisper-preprocessed
form `preprocessing/` produces — that one strips raw audio and tokenizes
against Whisper's vocab, which only serves ASR.

## Setup

```sh
uv sync
```

`main.py` handles the two one-time setup steps itself on first run — no
manual steps needed:

1. **Build the `monotonic_align` Cython extension** (native, platform-specific,
   not committed — rebuilds automatically on a new machine, e.g. the GPU box).
2. **Build the discriminator-equipped base checkpoint** for `--language-code`
   (default `quy`): `facebook/mms-tts-quy` ships generator-only (inference
   needs no discriminator); fine-tuning needs one. This downloads the
   original discriminator weights from `facebook/mms-tts` and merges them
   with the HF generator, caching the result at
   `finetune_hf_vits/mms-tts-quy-train/` (gitignored, ~300MB).

## Usage

```sh
uv run main.py
```

Pulls `--dataset` (default `gmezan/quechua-spanish-parallel`), fine-tunes for
`--epochs` (default 200), and saves the final model to `--output-dir`
(default `./vits-quy-finetuned`) — a plain `transformers.VitsModel` repo,
no PEFT/Coqui/vendored-code dependency needed to use it afterward.

For a quick correctness check instead of a real run:

```sh
uv run main.py --limit 8 --batch-size 2 --eval-batch-size 2 --epochs 1 --eval-steps 2
```

Run `uv run main.py --help` for the full list of options.

## Pushing to the Hub

```sh
uv run main.py --push-to-hub
```

`--hub-repo-id` defaults to `$HF_USERNAME/vits-quechua` (override with
`--hub-repo-id` directly if needed). Add `--private` to create the repo as
private. Needs write access via `HF_TOKEN` (see below). Remember the
finetuned model inherits MMS's non-commercial license.

## Auth: `.env` + `HF_TOKEN`/`HF_USERNAME`

A `.env` file already exists in this directory (gitignored via the repo's
root `.gitignore`) with:

```
HF_TOKEN=hf_...
HF_USERNAME=<your-hf-username>
```

`main.py` loads it automatically via `python-dotenv` at startup; `HF_TOKEN`
is also needed for the one-time base checkpoint build, which downloads from
the Hub. `datasets.load_dataset`, the Hub push, and the checkpoint download
all pick up `HF_TOKEN` from the environment with no explicit token argument
needed; `HF_USERNAME` is read directly to build the default `--dataset`/
`--hub-repo-id` values.

## Device: works on Mac, faster on CUDA

Unlike the earlier Coqui-based approach, this toolkit runs via `accelerate`
and works the same way on CPU, MPS, or CUDA — no code changes needed. `--fp16`
defaults to on when a CUDA GPU is available, off otherwise (mixed precision
isn't reliable on CPU/MPS). On this Mac that means CPU-only training — fine
for the `--limit`-capped smoke test above, far too slow for a real run. On
the Linux g5.xlarge box, just `uv sync` there and drop `--limit`.

## Compatibility patches

`finetune_hf_vits/` was written against `transformers~=4.35`. A few things
changed by `transformers` 5.x that this install uses (`stt/`/`preprocessing/`
also use transformers 5.x, so this stays consistent rather than pinning an
old version just for `tts/`):

- `datasets`' built-in audio decoding now requires `torchcodec` (needs an
  exact ffmpeg ABI match that isn't guaranteed across environments — the
  same issue `preprocessing/` hit). Patched `run_vits_finetuning.py` to
  decode audio manually via `soundfile`/`librosa` instead, same workaround
  used elsewhere in this repo.
- `TrainingArguments.logging_dir`/`group_by_length`/`overwrite_output_dir`
  were removed upstream; re-added `group_by_length`/`overwrite_output_dir`
  as fields on the script's own `VITSTrainingArguments`, and inlined a
  default for `logging_dir`.
- `PretrainedConfig`'s handling of `pad_token_id` changed; the vendored
  `VitsConfig` now sets it explicitly rather than relying on the base class.
- `VitsDiscriminator` never called `self.post_init()`, which newer
  `transformers.PreTrainedModel.from_pretrained()` requires unconditionally.
- `matplotlib`'s `FigureCanvasAgg.tostring_rgb()` was removed; alignment/
  spectrogram preview plots now use `buffer_rgba()`.

None of this changes the training math — see the diffs in
`finetune_hf_vits/` for exact changes (it's vendored source, not a
dependency, so they're plain edits, not a patch file).

## Verified

`uv sync`, `--help`, and a full tiny end-to-end run (`--limit 8`, 1 epoch,
CPU) — one-time Cython build, one-time base checkpoint conversion, training
with real GAN losses (disc/gen/kl/mel/duration/fmaps), mid-training and
final validation, and a saved checkpoint — all succeed.
