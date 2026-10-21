import argparse
import io
import os

import librosa
import soundfile as sf
from datasets import Audio, load_dataset
from dotenv import load_dotenv
from transformers import pipeline

from common import compute_error_rates, pick_device

load_dotenv()

HF_USERNAME = os.environ.get("HF_USERNAME", "gmezan")
DEFAULT_DATASET = f"{HF_USERNAME}/quechua-spanish-parallel"
DEFAULT_SPLIT = "train"
DEFAULT_AUDIO_COLUMN = "audio_quy"
DEFAULT_TEXT_COLUMN = "text_quy"
DEFAULT_MODEL = f"{HF_USERNAME}/whisper-quechua"
DEFAULT_EVAL_SIZE = 0.1
DEFAULT_SEED = 42
SAMPLING_RATE = 16000


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a fine-tuned Whisper STT model with WER/CER. Recovers "
            "the same held-out split stt/main.py trained against (same "
            "--eval-size/--seed, train_test_split over the same raw "
            "dataset), so this measures generalization, not memorized rows."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Raw HF dataset repo id (not the Whisper-ready one)")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--audio-column", default=DEFAULT_AUDIO_COLUMN)
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="STT model repo id to evaluate")
    parser.add_argument(
        "--language",
        default="spanish",
        help="Whisper decoder language anchor; must match what stt/ fine-tuned with",
    )
    parser.add_argument("--task", default="transcribe")
    parser.add_argument(
        "--eval-size",
        type=float,
        default=DEFAULT_EVAL_SIZE,
        help="Must match stt/main.py's --eval-size so the recovered split lines up with training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Must match stt/main.py's --seed so the recovered split lines up with training",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap eval rows for a quick smoke test")
    return parser.parse_args()


def load_eval_split(args):
    dataset = load_dataset(args.dataset, split=args.split)
    dataset = dataset.cast_column(args.audio_column, Audio(decode=False))
    eval_dataset = dataset.train_test_split(test_size=args.eval_size, seed=args.seed)["test"]
    if args.limit:
        eval_dataset = eval_dataset.select(range(min(args.limit, len(eval_dataset))))
    return eval_dataset


def decode_audio(raw_audio):
    source = io.BytesIO(raw_audio["bytes"]) if raw_audio.get("bytes") else raw_audio["path"]
    array, sr = sf.read(source, dtype="float32")
    if array.ndim > 1:
        array = array.mean(axis=1)
    if sr != SAMPLING_RATE:
        array = librosa.resample(array, orig_sr=sr, target_sr=SAMPLING_RATE)
    return array


def main():
    args = parse_args()
    device = pick_device()
    print(f"Device: {device}")

    print(f"1. Loading dataset ({args.dataset}, split={args.split})...")
    eval_dataset = load_eval_split(args)
    print(f"   {len(eval_dataset)} eval rows")

    print(f"2. Loading STT model ({args.model})...")
    asr = pipeline("automatic-speech-recognition", model=args.model, device=device)

    print("3. Transcribing eval set...")
    references, hypotheses = [], []
    for example in eval_dataset:
        reference = example[args.text_column]
        if not reference or not reference.strip():
            continue
        array = decode_audio(example[args.audio_column])
        result = asr(
            {"array": array, "sampling_rate": SAMPLING_RATE},
            generate_kwargs={"language": args.language, "task": args.task},
        )
        references.append(reference)
        hypotheses.append(result["text"])

    print("4. Computing WER/CER...")
    wer, cer = compute_error_rates(references, hypotheses)

    print(f"\nModel:   {args.model}")
    print(f"Dataset: {args.dataset} (split={args.split}, {len(references)} rows)")
    print(f"WER: {wer:.4f}")
    print(f"CER: {cer:.4f}")


if __name__ == "__main__":
    main()
