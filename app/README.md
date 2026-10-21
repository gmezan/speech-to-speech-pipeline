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
