import argparse
import os

from datasets import Audio, Features, Value, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

HF_USERNAME = os.environ.get("HF_USERNAME", "gmezan")
DEFAULT_SOURCE_DATASET = "josemercado/quechua-chanka-asr"
DEFAULT_SOURCE_SPLIT = "train"
DEFAULT_SOURCE_AUDIO_COLUMN = "audio"
DEFAULT_SOURCE_TEXT_COLUMN = "TEXTO"
DEFAULT_HUB_REPO_ID = f"{HF_USERNAME}/quechua-spanish-parallel"
SAMPLING_RATE = 16000

# See ../docs/dataset-schema.md. Columns with no source data yet (the Spanish
# side) are left blank (empty string / null) rather than omitted, so the
# schema is stable while the parallel collection effort fills them in.
SCHEMA = Features(
    {
        "id": Value("string"),
        "text_quy": Value("string"),
        "audio_quy": Audio(sampling_rate=SAMPLING_RATE),
        "text_spa": Value("string"),
        "audio_spa": Audio(sampling_rate=SAMPLING_RATE),
        "speaker_quy": Value("string"),
        "speaker_spa": Value("string"),
        "split": Value("string"),
    }
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap the parallel Quechua-Spanish dataset on the Hub "
            f"(creating it if needed) from the Quechua-only source "
            f"({DEFAULT_SOURCE_DATASET}), leaving the not-yet-collected "
            "Spanish side blank."
        )
    )
    parser.add_argument("--source-dataset", default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--source-split", default=DEFAULT_SOURCE_SPLIT)
    parser.add_argument("--source-audio-column", default=DEFAULT_SOURCE_AUDIO_COLUMN)
    parser.add_argument("--source-text-column", default=DEFAULT_SOURCE_TEXT_COLUMN)
    parser.add_argument(
        "--hub-repo-id",
        default=DEFAULT_HUB_REPO_ID,
        help="Target parallel-corpus dataset repo id",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hub repo as private if it doesn't exist yet",
    )
    parser.add_argument("--num-proc", type=int, default=4)
    return parser.parse_args()


def build_row_fn(audio_column, text_column):
    def to_row(example, idx):
        return {
            "id": f"quy-{idx:06d}",
            "text_quy": example[text_column],
            "audio_quy": example[audio_column],
            "text_spa": "",
            "audio_spa": None,
            "speaker_quy": None,
            "speaker_spa": None,
            "split": None,
        }

    return to_row


def main():
    args = parse_args()

    api = HfApi()
    if api.repo_exists(args.hub_repo_id, repo_type="dataset"):
        print(f"1. Dataset {args.hub_repo_id} already exists on the Hub, skipping creation.")
    else:
        print(f"1. Creating dataset repo {args.hub_repo_id} on the Hub...")
        api.create_repo(args.hub_repo_id, repo_type="dataset", private=args.private)

    print(f"2. Loading source dataset ({args.source_dataset}, split={args.source_split})...")
    source = load_dataset(args.source_dataset, split=args.source_split)
    source = source.cast_column(args.source_audio_column, Audio(decode=False))

    print("3. Mapping rows onto the parallel schema (Spanish side left blank)...")
    dataset = source.map(
        build_row_fn(args.source_audio_column, args.source_text_column),
        with_indices=True,
        remove_columns=source.column_names,
        features=SCHEMA,
        num_proc=args.num_proc,
    )

    print(f"4. Pushing to the Hub ({args.hub_repo_id})...")
    dataset.push_to_hub(args.hub_repo_id, private=args.private)

    print("Done! Dataset bootstrapped with the Quechua side populated and the Spanish side blank.")


if __name__ == "__main__":
    main()
