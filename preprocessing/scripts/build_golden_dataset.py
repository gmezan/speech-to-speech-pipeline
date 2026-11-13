"""Build the golden parallel Quechua-Spanish dataset from a preprocessing run's manifest.

Reads `manifiesto.csv` + its paired 16kHz audio directory (see
`preprocessing/docs/dataset-schema.md`) and emits a dataset with just the
columns a golden corpus needs: id, text_spa, text_quy, audio_spa, audio_quy.
No `source` or `split` column -- this is the ground-truth corpus splits get
derived from downstream, not a training-ready artifact itself.
"""

import argparse
import csv
import os

from datasets import Audio, Dataset, Features, Value
from dotenv import load_dotenv

DEFAULT_HUB_REPO_ID = "LosQuechuas/quechua-spanish-parallel-golden"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="../tmp/salida_preprocesamiento",
        help="Directory containing manifiesto.csv and audio_16k/{quy,spa}/",
    )
    parser.add_argument("--output-dir", default="./golden_dataset")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo-id", default=DEFAULT_HUB_REPO_ID)
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to a .env file to load HF_TOKEN from (defaults to preprocessing/.env)",
    )
    return parser.parse_args()


def load_manifest(input_dir):
    manifest_path = os.path.join(input_dir, "manifiesto.csv")
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    kept = []
    for row in rows:
        if row.get("incluido") != "True":
            continue
        audio_quy = os.path.join(input_dir, "audio_16k", "quy", f"{row['id']}.wav")
        audio_spa = os.path.join(input_dir, "audio_16k", "spa", f"{row['id']}.wav")
        if not os.path.isfile(audio_quy) or not os.path.isfile(audio_spa):
            raise FileNotFoundError(f"Missing audio for row {row['id']!r}: {audio_quy} / {audio_spa}")
        kept.append(
            {
                "id": row["id"],
                "text_spa": row["text_spa"],
                "text_quy": row["text_quy"],
                "audio_spa": audio_spa,
                "audio_quy": audio_quy,
            }
        )
    return kept


def main():
    args = parse_args()

    env_file = args.env_file or os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(env_file)

    rows = load_manifest(args.input_dir)
    print(f"Loaded {len(rows)} aligned pairs from {args.input_dir}")

    features = Features(
        {
            "id": Value("string"),
            "text_spa": Value("string"),
            "text_quy": Value("string"),
            "audio_spa": Audio(sampling_rate=16000),
            "audio_quy": Audio(sampling_rate=16000),
        }
    )
    ds = Dataset.from_list(rows, features=features)
    print(ds)

    ds.save_to_disk(args.output_dir)
    print(f"Saved golden dataset to {args.output_dir}")

    if args.push_to_hub:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN not set (check --env-file)")
        ds.push_to_hub(args.hub_repo_id, token=token, private=args.private)
        print(f"Pushed to https://huggingface.co/datasets/{args.hub_repo_id}")


if __name__ == "__main__":
    main()
