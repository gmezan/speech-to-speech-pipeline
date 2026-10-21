import argparse
import os
import warnings

import torch
from datasets import load_dataset
from dotenv import load_dotenv

# transformers' VITS modeling code still decorates a helper with the now-deprecated
# `torch.jit.script` (see transformers/models/vits/modeling_vits.py) — not our code,
# nothing to fix here, just silence the noise (same warning app/ silences).
warnings.filterwarnings("ignore", message="`torch.jit.script` is deprecated", category=FutureWarning)

from transformers import AutoTokenizer, VitsModel, pipeline  # noqa: E402

from common import compute_error_rates, pick_device

load_dotenv()

HF_USERNAME = os.environ.get("HF_USERNAME", "gmezan")
DEFAULT_DATASET = f"{HF_USERNAME}/quechua-spanish-parallel"
DEFAULT_SPLIT = "train"
DEFAULT_TEXT_COLUMN = "text_quy"
# facebook/mms-tts-quy until tts/ pushes a fine-tuned f"{HF_USERNAME}/vits-quechua"
DEFAULT_TTS_MODEL = "facebook/mms-tts-quy"
DEFAULT_STT_MODEL = f"{HF_USERNAME}/whisper-quechua"
DEFAULT_EVAL_SIZE = 0.1
DEFAULT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a TTS model via round-trip intelligibility: synthesize "
            "audio for each eval-split text, transcribe it with the fine-tuned "
            "STT model, and score WER/CER against the original text. This is "
            "an automatic intelligibility proxy, not a naturalness/MOS score "
            "(see README) — and it's bounded by the judge STT model's own "
            "error rate, not a pure TTS metric."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--tts-model", default=DEFAULT_TTS_MODEL, help="TTS model repo id to evaluate")
    parser.add_argument("--stt-model", default=DEFAULT_STT_MODEL, help="STT model repo id used as the judge")
    parser.add_argument(
        "--language",
        default="spanish",
        help="Whisper decoder language anchor for the judge model; must match what stt/ fine-tuned with",
    )
    parser.add_argument(
        "--eval-size",
        type=float,
        default=DEFAULT_EVAL_SIZE,
        help="Must match stt/main.py's --eval-size so text comes from the same held-out rows",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Must match stt/main.py's --seed so text comes from the same held-out rows",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap eval rows for a quick smoke test")
    return parser.parse_args()


def load_eval_texts(args):
    dataset = load_dataset(args.dataset, split=args.split)
    # Text-only: avoids decoding the audio columns (not needed here, and
    # requires torchcodec which isn't guaranteed to be installed).
    dataset = dataset.select_columns([args.text_column])
    eval_dataset = dataset.train_test_split(test_size=args.eval_size, seed=args.seed)["test"]
    if args.limit:
        eval_dataset = eval_dataset.select(range(min(args.limit, len(eval_dataset))))
    return [row[args.text_column] for row in eval_dataset if row[args.text_column] and row[args.text_column].strip()]


def main():
    args = parse_args()
    device = pick_device()
    print(f"Device: {device}")

    print(f"1. Loading dataset ({args.dataset}, split={args.split})...")
    texts = load_eval_texts(args)
    print(f"   {len(texts)} eval rows")

    print(f"2. Loading TTS model ({args.tts_model})...")
    tts_tokenizer = AutoTokenizer.from_pretrained(args.tts_model)
    tts_model = VitsModel.from_pretrained(args.tts_model).to(device)
    tts_model.eval()

    print(f"3. Loading STT judge model ({args.stt_model})...")
    asr = pipeline("automatic-speech-recognition", model=args.stt_model, device=device)

    print("4. Synthesizing and round-trip transcribing...")
    references, hypotheses, durations = [], [], []
    for text in texts:
        inputs = tts_tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            waveform = tts_model(**inputs).waveform
        audio = waveform.squeeze().cpu().numpy()
        sampling_rate = tts_model.config.sampling_rate
        durations.append(len(audio) / sampling_rate)

        result = asr({"array": audio, "sampling_rate": sampling_rate}, generate_kwargs={"language": args.language})
        references.append(text)
        hypotheses.append(result["text"])

    print("5. Computing round-trip WER/CER...")
    wer, cer = compute_error_rates(references, hypotheses)
    avg_duration = sum(durations) / len(durations) if durations else 0.0

    print(f"\nTTS model:  {args.tts_model}")
    print(f"STT judge:  {args.stt_model}")
    print(f"Dataset:    {args.dataset} (split={args.split}, {len(references)} rows)")
    print(f"Round-trip WER: {wer:.4f}")
    print(f"Round-trip CER: {cer:.4f}")
    print(f"Avg synthesized clip duration: {avg_duration:.2f}s")


if __name__ == "__main__":
    main()
