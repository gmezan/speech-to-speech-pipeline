import argparse
import os
import urllib.request

from datasets import Audio, Dataset, Features, Value
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

HF_USERNAME = os.environ.get("HF_USERNAME", "gmezan")
AMERICASNLP_BASE = (
    "https://raw.githubusercontent.com/AmericasNLP/americasnlp2024/master/"
    "ST1_MachineTranslation/data/quechua-spanish/parallel_data/es-quz"
)
DEFAULT_ES_URL = f"{AMERICASNLP_BASE}/jw300.es-quz.es"
DEFAULT_QUZ_URL = f"{AMERICASNLP_BASE}/jw300.es-quz.quz"
DEFAULT_SOURCE_VALUE = "americasnlp2024/jw300"
DEFAULT_HUB_REPO_ID = f"{HF_USERNAME}/quechua-spanish-parallel"
SAMPLING_RATE = 16000

# See ../docs/dataset-schema.md. jw300 is text-only parallel data -- the
# audio columns stay blank until real quz TTS/ASR recordings exist.
SCHEMA = Features(
    {
        "id": Value("string"),
        "text_quz": Value("string"),
        "audio_quz": Audio(sampling_rate=SAMPLING_RATE),
        "text_spa": Value("string"),
        "audio_spa": Audio(sampling_rate=SAMPLING_RATE),
        "speaker_quz": Value("string"),
        "speaker_spa": Value("string"),
        "source": Value("string"),
    }
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap the parallel Quechua-Spanish dataset on the Hub from "
            "AmericasNLP2024's genuine Quechua Collao (quz) JW300 parallel "
            "text -- not the quy/Chanka files the project used before. This "
            "REPLACES the dataset's contents (the old Chanka-era rows are "
            "being retired, not merged with); run "
            "preprocessing/scripts/assign_splits.py afterward to partition "
            "the result into train/validation/test."
        )
    )
    parser.add_argument("--es-url", default=DEFAULT_ES_URL)
    parser.add_argument("--quz-url", default=DEFAULT_QUZ_URL)
    parser.add_argument("--source-value", default=DEFAULT_SOURCE_VALUE)
    parser.add_argument("--hub-repo-id", default=DEFAULT_HUB_REPO_ID)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Download/build the dataset but don't push to the Hub")
    return parser.parse_args()


def fetch_lines(url):
    with urllib.request.urlopen(url) as resp:
        text = resp.read().decode("utf-8")
    return text.splitlines()


def main():
    args = parse_args()

    print(f"1. Downloading {args.es_url} ...")
    es_lines = fetch_lines(args.es_url)
    print(f"2. Downloading {args.quz_url} ...")
    quz_lines = fetch_lines(args.quz_url)
    if len(es_lines) != len(quz_lines):
        raise ValueError(
            f"Line count mismatch: {len(es_lines)} es lines vs {len(quz_lines)} quz lines -- "
            "these files are supposed to be sentence-aligned 1:1."
        )
    print(f"   {len(es_lines)} aligned sentence pairs")

    print("3. Building rows on the parallel schema (audio columns left blank)...")
    n = len(es_lines)
    rows = {
        "id": [f"quz-{i:06d}" for i in range(n)],
        "text_quz": quz_lines,
        "audio_quz": [None] * n,
        "text_spa": es_lines,
        "audio_spa": [None] * n,
        "speaker_quz": [None] * n,
        "speaker_spa": [None] * n,
        "source": [args.source_value] * n,
    }
    dataset = Dataset.from_dict(rows, features=SCHEMA)

    if args.dry_run:
        print("\nDry run -- not pushing to the Hub. Sample row:")
        print(dataset[0])
        return

    api = HfApi()
    if api.repo_exists(args.hub_repo_id, repo_type="dataset"):
        print(f"4. Dataset {args.hub_repo_id} already exists -- overwriting its contents.")
    else:
        print(f"4. Creating dataset repo {args.hub_repo_id} on the Hub...")
        api.create_repo(args.hub_repo_id, repo_type="dataset", private=args.private)

    print(f"5. Pushing to the Hub ({args.hub_repo_id})...")
    dataset.push_to_hub(args.hub_repo_id, private=args.private)

    print("Done! Run preprocessing/scripts/assign_splits.py next to partition into train/validation/test.")


if __name__ == "__main__":
    main()
