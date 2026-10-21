import os
import warnings

import gradio as gr
import librosa
import torch
from dotenv import load_dotenv

# transformers' VITS modeling code still decorates a helper with the now-deprecated
# `torch.jit.script` (see transformers/models/vits/modeling_vits.py) — not our code,
# nothing to fix here, just silence the noise.
warnings.filterwarnings("ignore", message="`torch.jit.script` is deprecated", category=FutureWarning)

from transformers import AutoTokenizer, VitsModel, pipeline  # noqa: E402

load_dotenv()

HF_USERNAME = os.environ.get("HF_USERNAME", "gmezan")
STT_MODEL_ID = f"{HF_USERNAME}/whisper-quechua"
TTS_MODEL_ID = "facebook/mms-tts-quy"
STT_LANGUAGE = "spanish"  # decoder language anchor used during fine-tuning (see preprocessing/)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()
print(f"Using device: {DEVICE}")

print(f"Loading STT model ({STT_MODEL_ID})...")
stt_pipeline = pipeline("automatic-speech-recognition", model=STT_MODEL_ID, device=DEVICE)

print(f"Loading TTS model ({TTS_MODEL_ID})...")
tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_ID)
tts_model = VitsModel.from_pretrained(TTS_MODEL_ID).to(DEVICE)
tts_model.eval()


def transcribe(audio_path):
    if not audio_path:
        return "No audio provided."

    # Decode and resample with librosa directly (bypasses ffmpeg_read issues on Mac).
    audio_array, _ = librosa.load(audio_path, sr=16000, mono=True)
    result = stt_pipeline(audio_array, generate_kwargs={"language": STT_LANGUAGE})
    return result["text"]


def synthesize(text):
    if not text or not text.strip():
        return None

    inputs = tts_tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        waveform = tts_model(**inputs).waveform

    audio = waveform.squeeze().cpu().numpy()
    return tts_model.config.sampling_rate, audio


with gr.Blocks(title="Quechua STT / TTS Playground") as demo:
    gr.Markdown("# Quechua Speech-to-Speech Playground")
    gr.Markdown(
        f"- **STT**: `{STT_MODEL_ID}` (Whisper, LoRA fine-tuned on Quechua)\n"
        f"- **TTS**: `{TTS_MODEL_ID}` (MMS VITS baseline, pending a custom fine-tune)"
    )

    with gr.Tab("Speech-to-Text"):
        stt_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Quechua audio")
        stt_button = gr.Button("Transcribe")
        stt_output = gr.Textbox(label="Transcription")
        stt_button.click(fn=transcribe, inputs=stt_audio, outputs=stt_output)

    with gr.Tab("Text-to-Speech"):
        tts_text = gr.Textbox(label="Quechua text", placeholder="Rimay qillqay...")
        tts_button = gr.Button("Synthesize")
        tts_output = gr.Audio(label="Synthesized audio")
        tts_button.click(fn=synthesize, inputs=tts_text, outputs=tts_output)


if __name__ == "__main__":
    demo.launch()
