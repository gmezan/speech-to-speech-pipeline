# app

Gradio playground (`playground.py`) to manually test STT, TTS, and
translation.

- **Speech-to-Text**: record or upload Quechua audio, transcribed by
  `$HF_USERNAME/whisper-quechua` (default
  [`gmezan/whisper-quechua`](https://huggingface.co/gmezan/whisper-quechua),
  the LoRA-fine-tuned Whisper model pushed by [`stt/`](../stt)).
- **Text-to-Speech**: type Quechua text, synthesized by
  [`facebook/mms-tts-quz`](https://huggingface.co/facebook/mms-tts-quz) — a
  baseline MMS VITS model, standing in until the custom fine-tune from
  [`tts/`](../tts) is ready.
- **Translation**: translate Spanish↔Quechua text with the fine-tuned NLLB
  model from [`nlp/translation/`](../nlp/translation) (default
  `$HF_USERNAME/nllb-quechua-spanish`, revision `epoch-3`).
- **Speech (SPA) -> Speech (QUZ)**: a basic chained pipeline — transcribes
  Spanish audio, translates it, and synthesizes the result — wiring together
  the Spanish STT, Translation, and TTS pieces above.

## Translation model and language codes

```
MT_MODEL_ID=<repo id, default $HF_USERNAME/nllb-quechua-spanish>
MT_MODEL_REVISION=<tag/branch/commit, default epoch-3, ignored for the base model>
```

Pass `MT_MODEL_ID=facebook/nllb-200-distilled-600M` to compare against the
untuned base checkpoint instead of the fine-tune.

## TTS model

```
TTS_MODEL_ID=<repo id, default facebook/mms-tts-quz>
```

Defaults to the MMS baseline since `$HF_USERNAME/vits-quechua` (`tts/`'s
fine-tuned output) hasn't been trained/pushed for real yet. Pass
`TTS_MODEL_ID=gmezan/vits-quechua` once that exists to switch to it.

The fine-tuned model was trained tagging Quechua as `quz_Latn` — not a real
pretrained NLLB language code (it resolves to the tokenizer's UNK token, not
a dedicated language embedding), but LoRA fine-tuning adapted consistently
around that signal, so the fine-tuned model expects the same tag at
inference. The untuned base checkpoint has no such learned association, so
`playground.py` automatically uses its real (if dialectally approximate)
code, `quy_Latn` (Ayacucho/Chanka Quechua), whenever `MT_MODEL_ID` points at
it — no manual flag needed, it's detected from the model id.

## Quechua-only — STT/TTS models do not do Spanish

The STT and TTS models above are Quechua-only, despite `stt/`'s
`--language spanish` flag: that flag is purely a Whisper decoder-token
anchor (Whisper has no dedicated Quechua token, so fine-tuning repurposes
the Spanish one to emit Quechua script — see
[`dataset-schema.md`](../preprocessing/docs/dataset-schema.md)), not a sign
the model was trained on any actual Spanish audio/text. `tts/`'s base
(`facebook/mms-tts-quz`) is likewise a dedicated monolingual Quechua MMS
checkpoint, never a Spanish one. (Translation is the exception — the NLLB
model above is genuinely bilingual.)

For real Spanish STT/TTS, the playground uses separate base models:

- **STT**: [`openai/whisper-small`](https://huggingface.co/openai/whisper-small)
  by default (override with `SPA_STT_MODEL_ID`), with `language="spanish"`.
- **TTS**: [`facebook/mms-tts-spa`](https://huggingface.co/facebook/mms-tts-spa),
  MMS's dedicated Spanish checkpoint (not currently wired into the
  playground's TTS tab, which is Quechua-only).

## Auth: `.env` + `HF_USERNAME`

Create a `.env` file in this directory (gitignored via the repo's root
`.gitignore`) with:

```
HF_USERNAME=<your-hf-username>
```

`playground.py` loads it automatically via `python-dotenv` at startup to
resolve the default model repo ids above. Add `HF_TOKEN=hf_...` too if any
of them are private.

## Usage

```sh
uv run playground.py
```

Launches a local Gradio UI (auto-detects `cuda` > `mps` > `cpu`).
