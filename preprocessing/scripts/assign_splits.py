import argparse
import hashlib
import os
from collections import Counter

from datasets import Audio, DatasetDict, load_dataset
from dotenv import load_dotenv

load_dotenv()

HF_USERNAME = os.environ.get("HF_USERNAME", "gmezan")
DEFAULT_DATASET = f"{HF_USERNAME}/quechua-spanish-parallel"
DEFAULT_SPLIT = "train"
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VALIDATION_RATIO = 0.1


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Partition the parallel Quechua-Spanish dataset into native "
            "train/validation/test HF splits -- shared by every model in "
            "this pipeline (translation, STT, TTS), so retraining any of "
            "them against the same --dataset always sees the same held-out "
            "rows. Each row's split is a deterministic hash of its own "
            "(text_quy, text_spa) content, not an index-based shuffle, so "
            "re-running this after adding new rows only assigns the new "
            "ones -- existing rows never move between splits. Literal "
            "duplicate sentence pairs (even under a different source/id) "
            "always land in the same split instead of leaking across train "
            "and test."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="HF split to load from the current (pre-partition) dataset")
    parser.add_argument("--text-column-quy", default="text_quy")
    parser.add_argument("--text-column-spa", default="text_spa")
    parser.add_argument("--source-column", default="source", help="Column to report the per-source breakdown by, if present")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--validation-ratio", type=float, default=DEFAULT_VALIDATION_RATIO)
    parser.add_argument("--hub-repo-id", default=None, help="Defaults to --dataset (push back in place)")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Report split counts/ratios without pushing to the Hub")
    args = parser.parse_args()
    if args.train_ratio + args.validation_ratio >= 1.0:
        parser.error("--train-ratio + --validation-ratio must leave room for a non-empty test split")
    return args


def assign_split(example, text_col_quy, text_col_spa, train_ratio, validation_ratio):
    quy = (example[text_col_quy] or "").strip().lower()
    spa = (example[text_col_spa] or "").strip().lower()
    digest = hashlib.sha256(f"{quy}|{spa}".encode("utf-8")).hexdigest()
    bucket = int(digest, 16) % 10_000 / 10_000  # deterministic, stable float in [0, 1)
    if bucket < train_ratio:
        split = "train"
    elif bucket < train_ratio + validation_ratio:
        split = "validation"
    else:
        split = "test"
    return {"_split": split}


def print_breakdown(label, dataset):
    counts = Counter(dataset["_split"])
    total = len(dataset)
    parts = ", ".join(f"{name}={counts.get(name, 0)} ({counts.get(name, 0) / total:.0%})" for name in ("train", "validation", "test"))
    print(f"   {label:35s} n={total:<7d} {parts}")


def main():
    args = parse_args()
    hub_repo_id = args.hub_repo_id or args.dataset

    print(f"1. Loading dataset ({args.dataset}, split={args.split})...")
    dataset = load_dataset(args.dataset, split=args.split)
    # Avoid decoding/re-encoding audio; keep the original bytes untouched.
    for col in ("audio_quy", "audio_spa"):
        if col in dataset.column_names:
            dataset = dataset.cast_column(col, Audio(decode=False))
    # Recomputed fresh (not read back) so this script stays correct even
    # against a dataset that never had a prior split assignment.
    if "split" in dataset.column_names:
        dataset = dataset.remove_columns(["split"])

    print("2. Assigning split per row (deterministic hash of text_quy/text_spa)...")
    dataset = dataset.map(
        lambda ex: assign_split(ex, args.text_column_quy, args.text_column_spa, args.train_ratio, args.validation_ratio)
    )

    print("3. Split breakdown:")
    print_breakdown("ALL", dataset)
    if args.source_column in dataset.column_names:
        for source in sorted(set(dataset[args.source_column])):
            print_breakdown(source, dataset.filter(lambda ex: ex[args.source_column] == source))

    splits = DatasetDict(
        {
            name: dataset.filter(lambda ex: ex["_split"] == name).remove_columns(["_split"])
            for name in ("train", "validation", "test")
        }
    )

    if args.dry_run:
        print("\nDry run -- not pushing to the Hub.")
        return

    print(f"4. Pushing to the Hub ({hub_repo_id}) as native train/validation/test splits...")
    splits.push_to_hub(hub_repo_id, private=args.private)
    print("Done!")


if __name__ == "__main__":
    main()
