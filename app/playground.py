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

from transformers import (  # noqa: E402
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    VitsModel,
    pipeline,
)

load_dotenv()

HF_USERNAME = os.environ.get("HF_USERNAME", "gmezan")
STT_MODEL_ID = f"{HF_USERNAME}/whisper-quechua"
TTS_MODEL_ID = f"{HF_USERNAME}/vits-quechua"
STT_LANGUAGE = "spanish"  # decoder language anchor used during fine-tuning (see preprocessing/)

# Base Whisper checkpoint for genuine Spanish speech (separate from the Quechua fine-tune above).
SPA_STT_MODEL_ID = os.environ.get("SPA_STT_MODEL_ID", "openai/whisper-small")

# Base NLLB-200 checkpoint, to be swapped for a SPA->QUY fine-tune later.
MT_MODEL_ID = os.environ.get("MT_MODEL_ID", "facebook/nllb-200-distilled-600M")
MT_LANG_CODES = {"Spanish": "spa_Latn", "Quechua": "quy_Latn"}


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

print(f"Loading Spanish STT model ({SPA_STT_MODEL_ID})...")
spa_stt_pipeline = pipeline("automatic-speech-recognition", model=SPA_STT_MODEL_ID, device=DEVICE)

print(f"Loading TTS model ({TTS_MODEL_ID})...")
tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_ID)
tts_model = VitsModel.from_pretrained(TTS_MODEL_ID).to(DEVICE)
tts_model.eval()

print(f"Loading MT model ({MT_MODEL_ID})...")
mt_tokenizer = AutoTokenizer.from_pretrained(MT_MODEL_ID)
mt_model = AutoModelForSeq2SeqLM.from_pretrained(MT_MODEL_ID).to(DEVICE)
mt_model.eval()


def transcribe(audio_path):
    if not audio_path:
        return "No audio provided."

    # Decode and resample with librosa directly (bypasses ffmpeg_read issues on Mac).
    audio_array, _ = librosa.load(audio_path, sr=16000, mono=True)
    result = stt_pipeline(audio_array, generate_kwargs={"language": STT_LANGUAGE})
    return result["text"]


def transcribe_spanish(audio_path):
    if not audio_path:
        return ""

    audio_array, _ = librosa.load(audio_path, sr=16000, mono=True)
    result = spa_stt_pipeline(audio_array, generate_kwargs={"language": "spanish"})
    return result["text"]


def synthesize(text):
    if not text or not text.strip():
        return None

    inputs = tts_tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        waveform = tts_model(**inputs).waveform

    audio = waveform.squeeze().cpu().numpy()
    return tts_model.config.sampling_rate, audio


def translate_text(text, src_name, tgt_name):
    if not text or not text.strip():
        return ""

    mt_tokenizer.src_lang = MT_LANG_CODES[src_name]
    inputs = mt_tokenizer(text, return_tensors="pt").to(DEVICE)
    forced_bos_token_id = mt_tokenizer.convert_tokens_to_ids(MT_LANG_CODES[tgt_name])
    with torch.no_grad():
        generated = mt_model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_new_tokens=256)

    return mt_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


def translate(text, direction):
    src_name, tgt_name = ("Spanish", "Quechua") if direction == "Spanish -> Quechua" else ("Quechua", "Spanish")
    return translate_text(text, src_name, tgt_name)


def speech_to_speech(audio_path):
    spanish_text = transcribe_spanish(audio_path)
    if not spanish_text.strip():
        return "No audio provided.", "", None

    quechua_text = translate_text(spanish_text, "Spanish", "Quechua")
    sample_rate, audio = synthesize(quechua_text)
    return spanish_text, quechua_text, (sample_rate, audio) if audio is not None else None


with gr.Blocks(title="Quechua STT / TTS Playground") as demo:
    gr.Markdown("# Quechua Speech-to-Speech Playground")
    gr.Markdown(
        f"- **STT**: `{STT_MODEL_ID}` (Whisper, LoRA fine-tuned on Quechua)\n"
        f"- **Spanish STT**: `{SPA_STT_MODEL_ID}` (base Whisper)\n"
        f"- **TTS**: `{TTS_MODEL_ID}` (MMS VITS baseline, pending a custom fine-tune)\n"
        f"- **MT**: `{MT_MODEL_ID}` (base NLLB-200, pending a SPA->QUY fine-tune)"
    )

    with gr.Tab("Speech (SPA) -> Speech (QUY)"):
        s2s_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Spanish audio")
        s2s_button = gr.Button("Translate speech")
        s2s_spanish_text = gr.Textbox(label="Transcribed Spanish text")
        s2s_quechua_text = gr.Textbox(label="Translated Quechua text")
        s2s_audio_output = gr.Audio(label="Synthesized Quechua audio")
        s2s_button.click(
            fn=speech_to_speech,
            inputs=s2s_audio,
            outputs=[s2s_spanish_text, s2s_quechua_text, s2s_audio_output],
        )

    with gr.Tab("Translation"):
        mt_direction = gr.Radio(
            ["Spanish -> Quechua", "Quechua -> Spanish"],
            value="Spanish -> Quechua",
            label="Direction",
        )
        mt_input = gr.Textbox(label="Input text", lines=4, placeholder="Escribe aquí...")
        mt_button = gr.Button("Translate")
        mt_output = gr.Textbox(label="Translation", lines=4)
        mt_button.click(fn=translate, inputs=[mt_input, mt_direction], outputs=mt_output)

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
