# app

Gradio playground to manually test the STT and TTS models in isolation (no
pipeline wiring between them yet).

- **Speech-to-Text**: record or upload Quechua audio, transcribed by
  `$HF_USERNAME/whisper-quechua` (default
  [`gmezan/whisper-quechua`](https://huggingface.co/gmezan/whisper-quechua),
  the LoRA-fine-tuned Whisper model pushed by [`stt/`](../stt)).
- **Text-to-Speech**: type Quechua text, synthesized by
  [`facebook/mms-tts-quy`](https://huggingface.co/facebook/mms-tts-quy) — a
  baseline MMS VITS model, standing in until the custom fine-tune from
  [`tts/`](../tts) is ready.

## Quechua-only — these models do not do Spanish

Both models above are Quechua-only, despite `stt/`'s `--language spanish`
flag: that flag is purely a Whisper decoder-token anchor (Whisper has no
dedicated Quechua token, so fine-tuning repurposes the Spanish one to emit
Quechua script — see
[`dataset-schema.md`](../preprocessing/docs/dataset-schema.md)), not a sign
the model was trained on any actual Spanish audio/text. `tts/`'s base
(`facebook/mms-tts-quy`) is likewise a dedicated monolingual Quechua MMS
checkpoint, never a Spanish one.

For real Spanish STT/TTS, use the unrelated base models instead of the
`gmezan/*` ones:

- **STT**: [`openai/whisper-small`](https://huggingface.co/openai/whisper-small)
  (or another multilingual Whisper checkpoint), with `language="spanish"`.
- **TTS**: [`facebook/mms-tts-spa`](https://huggingface.co/facebook/mms-tts-spa),
  MMS's dedicated Spanish checkpoint.

A combined Quechua+Spanish pipeline will need the Spanish side of the
parallel corpus collected first (see the root README's "What's left").

## Auth: `.env` + `HF_USERNAME`

Create a `.env` file in this directory (gitignored via the repo's root
`.gitignore`) with:

```
HF_USERNAME=<your-hf-username>
```

`main.py` loads it automatically via `python-dotenv` at startup to resolve
the STT model repo id above. Add `HF_TOKEN=hf_...` too if that repo is
private.

## Usage

```sh
uv run main.py
```

Launches a local Gradio UI (auto-detects `cuda` > `mps` > `cpu`).
