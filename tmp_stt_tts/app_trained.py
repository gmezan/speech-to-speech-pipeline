import gradio as gr
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from peft import PeftModel
from TTS.api import TTS
import librosa
import numpy as np

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Load Quechua-Fine-Tuned Whisper
print("Loading Quechua Whisper STT...")
base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
peft_model = PeftModel.from_pretrained(base_model, "./whisper-quechua-lora-final")
# Merge LoRA weights into base model for fast inference
merged_stt_model = peft_model.merge_and_unload()

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="spanish", task="transcribe")

stt_pipeline = pipeline(
    "automatic-speech-recognition",
    model=merged_stt_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device
)

# 2. Load TTS (Spanish baseline for now)
print("Loading TTS...")
tts_model = TTS(model_name="tts_models/es/css10/vits", progress_bar=False).to(device)

def mock_nlp(text):
    return f"[Quechua Processed]: {text}"

def process_audio(audio_path):
    if not audio_path:
        return None, "No audio provided."
        
    # 1. Load and resample audio array directly with librosa (bypasses ffmpeg_read)
    audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
    
    # 2. STT via fine-tuned Whisper (pass the numpy array directly)
    stt_result = stt_pipeline(audio_array, generate_kwargs={"language": "spanish"})
    transcribed_text = stt_result["text"]
    
    # 3. Mock NLP
    response_text = mock_nlp(transcribed_text)
    
    # 4. TTS Output
    output_audio_path = "output.wav"
    tts_model.tts_to_file(text=response_text, file_path=output_audio_path)
    
    return output_audio_path, response_text

interface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(sources=["microphone"], type="filepath", label="Speak Quechua or Spanish"),
    outputs=[
        gr.Audio(label="Audio Output"),
        gr.Textbox(label="Quechua Transcription")
    ],
    title="Quechua STT -> Mock NLP -> TTS Pipeline"
)

if __name__ == "__main__":
    interface.launch()