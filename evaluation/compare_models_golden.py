import argparse
import os

from dotenv import load_dotenv

import eval_golden
import eval_mt

load_dotenv()

HF_USERNAME = os.environ.get("HF_USERNAME", "gmezan")
DEFAULT_RESULTS_PATH = "golden_results.jsonl"
DEFAULT_SUMMARY_PATH = "golden_summary.md"

# label=repo_id[@revision] -- the base (untuned) model plus every epoch tagged by
# nlp/translation/train.py's --tag, so each session's progress (or regression) on this
# out-of-domain medical set is visible side by side, not just the in-domain test scores.
DEFAULT_MODEL_SPECS = [
    "base=facebook/nllb-200-distilled-600M",
    f"epoch-1={HF_USERNAME}/nllb-quechua-spanish@epoch-1",
    f"epoch-2={HF_USERNAME}/nllb-quechua-spanish@epoch-2",
    f"epoch-3={HF_USERNAME}/nllb-quechua-spanish@epoch-3",
]


def parse_model_spec(spec):
    label, rest = spec.split("=", 1)
    model, _, revision = rest.partition("@")
    return label, model, (revision or None)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run evaluation/eval_golden.py's medical-domain golden-set scoring across "
            "several models/revisions in one pass, and write a single comparison table -- "
            "so you can see at a glance how the base model and each fine-tuned epoch "
            "generalize out-of-domain, instead of running eval_golden.py once per model "
            "and eyeballing separate outputs."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODEL_SPECS,
        help="label=repo_id[@revision] entries to compare, e.g. epoch-3=gmezan/nllb-quechua-spanish@epoch-3",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap eval rows per direction per model, for a quick smoke test")
    parser.add_argument("--results-path", default=DEFAULT_RESULTS_PATH, help="JSONL file every model's run appends its scores to")
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH, help="Markdown file the final comparison table is written to")
    return parser.parse_args()


def build_summary_table(all_results):
    """One row per (model, direction), the `__all__` bucket only -- the golden set has
    no `source` column to break down further, so per-source and __all__ are identical
    anyway (see eval_mt.py's bucket logic)."""
    lines = ["| model | direction | n | BLEU | chrF++ |", "|---|---|---:|---:|---:|"]
    for label, results in all_results:
        for direction, by_source in results["directions"].items():
            scores = by_source.get("__all__")
            if not scores:
                continue
            lines.append(f"| {label} | `{direction}` | {scores['n']} | {scores['bleu']:.2f} | {scores['chrf++']:.2f} |")
    return "\n".join(lines)


def main():
    args = parse_args()

    try:
        import torch
    except ImportError:
        torch = None

    all_results = []
    for spec in args.models:
        label, model, revision = parse_model_spec(spec)
        print(f"\n=== {label} ({model}" + (f", revision={revision}" if revision else "") + ") ===")
        eval_args = eval_golden.build_args(
            model=model,
            revision=revision,
            limit=args.limit,
            results_path=args.results_path,
        )
        results = eval_mt.run(eval_args)
        all_results.append((label, results))

        # Each iteration loads a fresh ~600M-param model; free GPU memory before the next one.
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    table = build_summary_table(all_results)
    print("\n" + table)

    with open(args.summary_path, "w") as f:
        f.write("# Golden-set (medical domain) model comparison\n\n")
        f.write(f"Dataset: `{eval_golden.DEFAULT_DATASET}` (split=`{eval_golden.DEFAULT_SPLIT}`)\n\n")
        f.write(table + "\n")
    print(f"\nWrote summary to {args.summary_path}")


if __name__ == "__main__":
    main()
