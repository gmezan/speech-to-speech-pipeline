# Hardware requirements

Illustrative disk/RAM/GPU sizing for every model this pipeline uses or
produces, plus totals for two deployment shapes: **Quechua-only** vs.
**both languages** (adds Spanish STT/TTS + translation). Numbers are
directional, not benchmarked on this project's actual hardware — treat
them as "what tier of box do I need," not exact figures.

## Method & assumptions

- **Disk** = actual Hub repo checkpoint size (confirmed via the HF Hub API
  for each base model below), fp32 and fp16 variants. A fine-tune (LoRA,
  merged) is the same architecture as its base, so its disk size is the
  same as the base's.
- **RAM (inference)** = rule of thumb of **~2× the fp16 checkpoint size**,
  to cover the loaded weights plus Python/`transformers`/tokenizer/audio
  preprocessing overhead. Running fp32 on CPU roughly doubles this again.
- **GPU VRAM (inference)** = fp16 weights + activation/KV-cache overhead
  for a single request at a time; batching multiple requests scales this
  up.
- **GPU VRAM (LoRA fine-tune)** = frozen base weights (fp16) + LoRA
  adapter params/gradients/optimizer state (small — a few million params)
  + activation memory, which scales with batch size and sequence/audio
  length. This project only does LoRA fine-tuning (see `stt/`, `tts/`,
  `nlp/translation/`), never a full fine-tune, so full-fine-tune VRAM
  (which would need to hold optimizer state for *every* parameter, not
  just the adapter) isn't tabulated.
- All training in this repo happens **sequentially**, one model at a time,
  on a single GPU box (`infra/aws_ec2.sh` provisions a `g5.xlarge` — 1×
  NVIDIA A10G, 24GB VRAM, 100GB EBS, 16GB host RAM) — so the training
  totals below are a **max across models**, not a sum.

## Per-model sizing

### STT — Whisper (`openai/whisper-small`, 244M params)

| variant | repo | disk (fp32 / fp16) | RAM (inference) | GPU VRAM (inference) | GPU VRAM (LoRA fine-tune) |
|---|---|---|---|---|---|
| Base (Spanish, untuned) | `openai/whisper-small` | 967MB / 484MB | ~1GB | ~1–2GB | ~6–10GB |
| Fine-tuned (Quechua) | `gmezan/whisper-quechua` | 967MB / 484MB | ~1GB | ~1–2GB | ~6–10GB |

Fine-tuning merges a LoRA adapter back into the base, so the pushed model
is byte-for-byte the same architecture/size as the base — only the
weights differ.

### TTS — MMS VITS (monolingual, ~36M params)

| variant | repo | disk (fp32 / fp16) | RAM (inference) | GPU VRAM (inference) | GPU VRAM (fine-tune) |
|---|---|---|---|---|---|
| Base (Spanish) | `facebook/mms-tts-spa` | 145MB / 73MB | <1GB | <1GB | ~8–12GB |
| Base (Quechua, today's app fallback) | `facebook/mms-tts-quy` | 145MB / 73MB | <1GB | <1GB | ~8–12GB |
| Fine-tuned (Quechua, not pushed yet) | `gmezan/vits-quechua` | 145MB / 73MB | <1GB | <1GB | ~8–12GB |

The checkpoint itself is tiny, but fine-tuning (via the vendored
`ylacombe/finetune-hf-vits`) also trains a GAN discriminator alongside the
generator and operates on raw waveforms (long sequences), which is why
training VRAM is disproportionately higher than the model's own size —
consistent with this project's own CPU smoke test needing a small
`--limit`/batch size to stay fast.

### Translation — NLLB-200 (`facebook/nllb-200-distilled-600M`, 600M params)

| variant | repo | disk (fp32 / fp16) | RAM (inference) | GPU VRAM (inference) | GPU VRAM (LoRA fine-tune) |
|---|---|---|---|---|---|
| Base (both directions, untuned) | `facebook/nllb-200-distilled-600M` | 2.46GB / 1.23GB | ~2.5GB | ~2–4GB | ~10–14GB |
| Fine-tuned (Quechua↔Spanish, not pushed yet) | `gmezan/nllb-quechua-spanish` | 2.46GB / 1.23GB | ~2.5GB | ~2–4GB | ~10–14GB |

One model handles both translation directions (see
[`nlp/README.md`](../nlp/README.md)), so there's no separate per-direction
variant.

## Grouped totals

### A. Quechua-only deployment (no translation)

Serving `app/` with just `gmezan/whisper-quechua` + `gmezan/vits-quechua`
(or today's `facebook/mms-tts-quy` baseline) loaded at once:

| | disk (fp16) | RAM | GPU VRAM (inference) |
|---|---|---|---|
| **Total** | ~557MB | ~1.5–2GB | ~2–3GB |

Fits comfortably on a single modest GPU (e.g. a T4, 16GB) or even CPU-only
for low-traffic use, since neither model alone needs much VRAM.

### B. Both-languages deployment (full Quechua↔Spanish speech-to-speech)

Serving all five models concurrently — `gmezan/whisper-quechua`,
`openai/whisper-small` (Spanish STT), `gmezan/vits-quechua`,
`facebook/mms-tts-spa`, `gmezan/nllb-quechua-spanish`:

| | disk (fp16) | RAM | GPU VRAM (inference) |
|---|---|---|---|
| **Total** | ~2.3GB | ~7–8GB | ~6–10GB |

Still fits on a single mid-range GPU (e.g. a 16–24GB card); lazy-loading
models per request instead of keeping all five resident would lower this
further at the cost of latency on cold requests.

### Training (sequential — one model at a time)

| stage | peak GPU VRAM | fits on `g5.xlarge` (24GB)? |
|---|---|---|
| `stt/` (Whisper LoRA) | ~6–10GB | yes, with headroom |
| `tts/` (VITS fine-tune) | ~8–12GB | yes, with headroom |
| `nlp/translation/` (NLLB LoRA) | ~10–14GB | yes, with headroom |

The current `g5.xlarge` choice in `infra/aws_ec2.sh` (A10G, 24GB VRAM,
100GB EBS) comfortably covers every training stage in this pipeline,
including the not-yet-run translation fine-tune — no instance-type change
needed to add translation into the training rotation.
