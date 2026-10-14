import argparse
import io

import librosa
import soundfile as sf
from datasets import Audio, load_dataset
from dotenv import load_dotenv
from transformers import WhisperProcessor

DEFAULT_DATASET = "josemercado/quechua-chanka-asr"
DEFAULT_SPLIT = "train"
DEFAULT_AUDIO_COLUMN = "audio"
DEFAULT_TEXT_COLUMN = "TEXTO"
DEFAULT_MODEL = "openai/whisper-small"
DEFAULT_OUTPUT_DIR = "./quechua_whisper_ready"
SAMPLING_RATE = 16000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a Quechua ASR dataset for Whisper fine-tuning."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset repo id")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--audio-column", default=DEFAULT_AUDIO_COLUMN)
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Whisper checkpoint to size the processor")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Also push the processed dataset to the Hub (needs HF_TOKEN, e.g. via a .env file)",
    )
    parser.add_argument(
        "--hub-repo-id",
        help="Target dataset repo id, e.g. 'myorg/quechua-whisper-ready'. Required with --push-to-hub",
    )
    parser.add_argument("--private", action="store_true", help="Create the Hub repo as private")
    parser.add_argument(
        "--language",
        default="spanish",
        help=(
            "Whisper decoder language token. Quechua has no dedicated Whisper "
            "language token, so we anchor to a related supported language and "
            "let fine-tuning adapt the output script."
        ),
    )
    parser.add_argument("--task", default="transcribe")
    args = parser.parse_args()
    if args.push_to_hub and not args.hub_repo_id:
        parser.error("--hub-repo-id is required with --push-to-hub")
    return args


def build_prepare_fn(processor, audio_column, text_column):
    def prepare_dataset(batch):
        audio = batch[audio_column]
        source = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
        array, sr = sf.read(source, dtype="float32")
        if array.ndim > 1:
            array = array.mean(axis=1)
        if sr != SAMPLING_RATE:
            array = librosa.resample(array, orig_sr=sr, target_sr=SAMPLING_RATE)

        batch["input_features"] = processor.feature_extractor(
            array, sampling_rate=SAMPLING_RATE
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch[text_column]).input_ids
        return batch

    return prepare_dataset


def main():
    load_dotenv()
    args = parse_args()

    print(f"1. Loading Whisper processor ({args.model})...")
    processor = WhisperProcessor.from_pretrained(
        args.model, language=args.language, task=args.task
    )

    print(f"2. Loading dataset ({args.dataset}, split={args.split})...")
    dataset = load_dataset(args.dataset, split=args.split)

    print("3. Switching audio column to raw (undecoded) bytes...")
    dataset = dataset.cast_column(args.audio_column, Audio(decode=False))

    print(f"4. Decoding audio, resampling to {SAMPLING_RATE}Hz, extracting features, and tokenizing labels...")
    prepared_dataset = dataset.map(
        build_prepare_fn(processor, args.audio_column, args.text_column),
        remove_columns=dataset.column_names,
        num_proc=args.num_proc,
    )

    print(f"5. Saving processed dataset to {args.output_dir}...")
    prepared_dataset.save_to_disk(args.output_dir)

    if args.push_to_hub:
        print(f"6. Pushing processed dataset to the Hub ({args.hub_repo_id})...")
        prepared_dataset.push_to_hub(args.hub_repo_id, private=args.private)

    print("Done! Dataset is ready for training.")


if __name__ == "__main__":
    main()
