import argparse
import os
from collections import Counter

from datasets import DatasetDict, load_dataset
from dotenv import load_dotenv

load_dotenv()

HF_USERNAME = os.environ.get("HF_USERNAME", "gmezan")
DEFAULT_DATASET = f"{HF_USERNAME}/quechua-spanish-parallel"
DEFAULT_SOURCE_PREFIX = "americasnlp"

# Characters NLLB's SentencePiece vocab has no entry for (verified against
# facebook/nllb-200-distilled-600M -- they decode to <unk>, an outright
# information loss, not byte-fallback), plus invisible zero-width
# characters. Same map evaluation/eval_mt.py normalizes with before scoring;
# applied here at the data level so fine-tuning doesn't learn from targets
# that collapse to <unk>.
UNK_CHAR_MAP = {
    "“": '"', "”": '"',
    "‘": "'", "’": "'",
    "—": "-", "–": "-",
    "​": "",
    "⁠": "",
}


def normalize_text(text):
    if not text:
        return text
    for src, dst in UNK_CHAR_MAP.items():
        text = text.replace(src, dst)
    return text


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Clean text_quy/text_spa of characters that decode to NLLB's "
            "<unk> token (curly quotes, em/en dashes) plus invisible "
            "zero-width characters. Scoped to rows whose --source-column "
            "starts with --source-prefix (default 'americasnlp') -- the "
            "only sources this noise was found in (see docs/datasets/"
            "quechua-spanish-parallel.md); josemercado/quechua-chanka-asr "
            "is left untouched."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--text-column-quy", default="text_quy")
    parser.add_argument("--text-column-spa", default="text_spa")
    parser.add_argument("--source-column", default="source")
    parser.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX, help="Only rows whose source starts with this are cleaned")
    parser.add_argument("--hub-repo-id", default=None, help="Defaults to --dataset (push back in place)")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Report how many rows would change without pushing")
    return parser.parse_args()


def build_clean_fn(text_col_quy, text_col_spa, source_col, source_prefix, changed_counter):
    def _clean(ex):
        if not str(ex.get(source_col, "")).startswith(source_prefix):
            return ex
        new_quy = normalize_text(ex[text_col_quy])
        new_spa = normalize_text(ex[text_col_spa])
        if new_quy != ex[text_col_quy] or new_spa != ex[text_col_spa]:
            changed_counter[ex[source_col]] += 1
            ex = dict(ex)
            ex[text_col_quy] = new_quy
            ex[text_col_spa] = new_spa
        return ex

    return _clean


def main():
    args = parse_args()
    hub_repo_id = args.hub_repo_id or args.dataset

    print(f"1. Loading dataset ({args.dataset}, all splits)...")
    dataset = load_dataset(args.dataset)

    cleaned_splits = {}
    for split_name, split in dataset.items():
        print(f"2. Cleaning '{split_name}' split ({len(split)} rows)...")
        changed = Counter()
        cleaned_splits[split_name] = split.map(
            build_clean_fn(args.text_column_quy, args.text_column_spa, args.source_column, args.source_prefix, changed)
        )
        if changed:
            for source, n in changed.most_common():
                print(f"   {source:35s} {n} rows changed")
        else:
            print("   0 rows changed")

    if args.dry_run:
        print("\nDry run -- not pushing to the Hub.")
        return

    print(f"3. Pushing to the Hub ({hub_repo_id})...")
    DatasetDict(cleaned_splits).push_to_hub(hub_repo_id, private=args.private)
    print("Done!")


if __name__ == "__main__":
    main()
