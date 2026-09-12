import gradio as gr
import torch
from transformers import pipeline
from TTS.api import TTS
import scipy.io.wavfile as wavfile

# 1. Determine Mac Hardware Acceleration
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Load Whisper (STT)
print("Loading Whisper...")
stt_model = pipeline(
    "automatic-speech-recognition", 
    model="openai/whisper-small", 
    device=device
)

# 3. Load Coqui TTS (Spanish VITS model)
print("Loading TTS...")
tts_model = TTS(model_name="tts_models/es/css10/vits", progress_bar=False).to(device)

# 4. Mock NLP Model
def mock_nlp(text):
    # Eventually, your translation/LLM logic goes here.
    # For now, it just echoes the text to prove the pipeline works.
    return text

# 5. Pipeline Logic
def process_audio(audio_path):
    if not audio_path:
        return None, "No audio provided."
        
    # Step A: Speech to Text
    stt_result = stt_model(audio_path, generate_kwargs={"language": "spanish"})
    transcribed_text = stt_result["text"]
    
    # Step B: NLP (Mock)
    response_text = mock_nlp(transcribed_text)
    
    # Step C: Text to Speech
    output_audio_path = "output.wav"
    tts_model.tts_to_file(text=response_text, file_path=output_audio_path)
    
    return output_audio_path, response_text

# 6. Build the UI
interface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(sources=["microphone"], type="filepath", label="Speak Spanish here"),
    outputs=[
        gr.Audio(label="AI Voice Response"),
        gr.Textbox(label="Transcribed Text")
    ],
    title="STT -> NLP -> TTS Pipeline (Spanish Baseline)",
    description="Speak into the microphone in Spanish. It will transcribe, process, and speak it back."
)

if __name__ == "__main__":
    interface.launch()