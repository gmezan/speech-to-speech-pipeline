ssh -i *.pem ubuntu@<instance_public_ip>

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

git clone https://github.com/gmezan/speech-to-speech-pipeline.git
cd stt
uv venv --python 3.11
source .venv/bin/activate
uv sync
export HF_TOKEN=<your_huggingface_token>
export HF_USERNAME=<your_huggingface_username>

# STT (pushes to $HF_USERNAME/whisper-quechua)
uv run main.py --push-to-hub

# TTS (pushes to $HF_USERNAME/vits-quechua)
uv run main.py --push-to-hub
