import argparse
import json
import os
import time
from collections import defaultdict

import sacrebleu
from datasets import concatenate_datasets, load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from common import pick_device

load_dotenv()

HF_USERNAME = os.environ.get("HF_USERNAME", "gmezan")
DEFAULT_DATASET = f"{HF_USERNAME}/quechua-spanish-parallel"
DEFAULT_SPLIT = "test"
DEFAULT_MODEL = f"{HF_USERNAME}/nllb-quechua-spanish"
DEFAULT_MAX_LENGTH = 128
DEFAULT_RESULTS_PATH = "mt_results.jsonl"

# Same FLORES-200 codes as translation/common.py (quy_Latn = Ayacucho
# Quechua, the closest NLLB-supported code to this project's Chanka variant).
LANG_CODES = {"quy": "quy_Latn", "spa": "spa_Latn"}

# Characters NLLB's SentencePiece vocab has no entry for (verified
# empirically against facebook/nllb-200-distilled-600M -- they decode to
# <unk>, an outright information loss, not byte-fallback) plus invisible
# zero-width characters. Normalizing both hypothesis and reference before
# scoring means this kind of formatting noise -- not present in the corpus
# because of any translation quality issue -- doesn't get counted as one.
UNK_CHAR_MAP = {
    "“": '"', "”": '"',  # “ ”
    "‘": "'", "’": "'",  # ‘ ’
    "—": "-", "–": "-",  # — –
    "​": "", "⁠": "",  # zero-width space / word joiner
}


def normalize_text(text):
    for src, dst in UNK_CHAR_MAP.items():
        text = text.replace(src, dst)
    return text


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an NLLB quy<->spa translation model with BLEU/chrF++. "
            "Loads the native 'test' split (see "
            "preprocessing/scripts/assign_splits.py) -- a fixed benchmark "
            "assigned once per row by content hash, so it stays the same set "
            "regardless of how much the dataset grows later, and "
            "translation/train.py never loads it, so no training decision "
            "can have indirectly optimized against it."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Raw HF dataset repo id (text_quy/text_spa columns)")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Native HF split to score on")
    parser.add_argument("--text-column-quy", default="text_quy")
    parser.add_argument("--text-column-spa", default="text_spa")
    parser.add_argument("--source-column", default="source", help="Column to break scores down by, if present")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Translation model repo id or local path to evaluate")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Cap eval rows per direction for a quick smoke test")
    parser.add_argument("--results-path", default=DEFAULT_RESULTS_PATH, help="JSONL file to append this run's scores to")
    return parser.parse_args()


def build_eval_split(args):
    """Loads the native --split (a value fixed per row by
    preprocessing/scripts/assign_splits.py, a deterministic hash of each
    row's own text content) before building the bidirectional (quy->spa /
    spa->quy) examples -- so both directions of a given sentence pair always
    share the same split, and this benchmark set doesn't change just because
    the dataset grows. --source-column rides along only for reporting."""
    dataset = load_dataset(args.dataset, split=args.split)
    columns = [args.text_column_quy, args.text_column_spa]
    has_source = args.source_column in dataset.column_names
    if has_source:
        columns.append(args.source_column)
    dataset = dataset.select_columns(columns)

    def tag(src_col, tgt_col, src_lang, tgt_lang):
        def _fn(ex):
            row = {
                "src_text": ex[src_col],
                "tgt_text": ex[tgt_col],
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
            }
            if has_source:
                row["source"] = ex[args.source_column]
            return row

        return _fn

    quy_to_spa = dataset.map(
        tag(args.text_column_quy, args.text_column_spa, LANG_CODES["quy"], LANG_CODES["spa"]),
        remove_columns=dataset.column_names,
    )
    spa_to_quy = dataset.map(
        tag(args.text_column_spa, args.text_column_quy, LANG_CODES["spa"], LANG_CODES["quy"]),
        remove_columns=dataset.column_names,
    )
    combined = concatenate_datasets([quy_to_spa, spa_to_quy])
    return combined.filter(lambda ex: ex["src_text"] and ex["tgt_text"])


def translate_batch(tokenizer, model, device, texts, src_lang, tgt_lang, max_length):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    generated = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_length=max_length)
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def main():
    args = parse_args()
    device = pick_device()
    print(f"Device: {device}")

    print(f"1. Loading '{args.split}' split ({args.dataset})...")
    eval_dataset = build_eval_split(args)
    # Rows are stored in contiguous per-source blocks (see `source` column),
    # so --limit would otherwise only ever sample the first source instead
    # of a representative slice -- shuffle first, deterministically.
    eval_dataset = eval_dataset.shuffle(seed=42)
    print(f"   {len(eval_dataset)} eval rows (both directions combined)")

    print(f"2. Loading translation model ({args.model})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    model.eval()

    print("3. Translating eval set...")
    rows_by_direction = defaultdict(list)
    for ex in eval_dataset:
        rows_by_direction[(ex["src_lang"], ex["tgt_lang"])].append(ex)

    # direction -> source -> {"refs": [...], "hyps": [...]}; "__all__" is the
    # combined bucket for that direction regardless of source.
    buckets = defaultdict(lambda: defaultdict(lambda: {"refs": [], "hyps": []}))

    start = time.time()
    n_translated = 0
    for (src_lang, tgt_lang), rows in rows_by_direction.items():
        if args.limit:
            rows = rows[: args.limit]
        direction = f"{src_lang}->{tgt_lang}"
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i : i + args.batch_size]
            hyps = translate_batch(
                tokenizer, model, device,
                [r["src_text"] for r in batch],
                src_lang, tgt_lang, args.max_length,
            )
            for row, hyp in zip(batch, hyps):
                source = row.get("source", "unknown")
                ref = normalize_text(row["tgt_text"])
                hyp = normalize_text(hyp)
                buckets[direction][source]["refs"].append(ref)
                buckets[direction][source]["hyps"].append(hyp)
                buckets[direction]["__all__"]["refs"].append(ref)
                buckets[direction]["__all__"]["hyps"].append(hyp)
            n_translated += len(batch)
    elapsed = time.time() - start
    rate = n_translated / elapsed if elapsed else None
    print(f"   Translated {n_translated} sentences in {elapsed:.1f}s" + (f" ({rate:.1f} sent/s)" if rate else ""))

    print("4. Scoring (BLEU + chrF++)...")
    results = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "sentences_per_second": round(rate, 2) if rate else None,
        "directions": {},
    }
    for direction, by_source in buckets.items():
        results["directions"][direction] = {}
        for source, data in by_source.items():
            if not data["refs"]:
                continue
            bleu = sacrebleu.corpus_bleu(data["hyps"], [data["refs"]])
            chrf = sacrebleu.corpus_chrf(data["hyps"], [data["refs"]], word_order=2)
            results["directions"][direction][source] = {
                "n": len(data["refs"]),
                "bleu": round(bleu.score, 2),
                "chrf++": round(chrf.score, 2),
            }

    print(f"\nModel:   {args.model}")
    print(f"Dataset: {args.dataset} (split={args.split})")
    for direction, by_source in results["directions"].items():
        print(f"\n  {direction}")
        for source, scores in sorted(by_source.items(), key=lambda kv: kv[0] != "__all__"):
            label = "ALL" if source == "__all__" else source
            print(f"    {label:35s} n={scores['n']:<6d} BLEU={scores['bleu']:<6.2f} chrF++={scores['chrf++']:.2f}")

    with open(args.results_path, "a") as f:
        record = {**results, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        f.write(json.dumps(record) + "\n")
    print(f"\nAppended results to {args.results_path}")


if __name__ == "__main__":
    main()
