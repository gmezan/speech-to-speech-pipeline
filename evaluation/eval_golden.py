import argparse

import eval_mt

DEFAULT_DATASET = "LosQuechuas/quechua-spanish-parallel-golden"
DEFAULT_SPLIT = "train"
# This external dataset predates the project's quy->quz rename and still uses the old
# column name -- the *content* is genuine Collao Quechua, just under the old header.
DEFAULT_TEXT_COLUMN_QUZ = "text_quy"
DEFAULT_TEXT_COLUMN_SPA = "text_spa"
DEFAULT_RESULTS_PATH = "golden_results.jsonl"


def build_args(model, revision=None, limit=None, results_path=DEFAULT_RESULTS_PATH, batch_size=16, max_length=eval_mt.DEFAULT_MAX_LENGTH):
    """Builds the args namespace eval_mt.run() expects, pre-filled with this golden
    dataset's shape -- the single knobs callers (this script's CLI, or
    compare_models_golden.py looping over several models) need are which model/revision
    to score and how to cap/log the run."""
    return argparse.Namespace(
        dataset=DEFAULT_DATASET,
        split=DEFAULT_SPLIT,
        text_column_quz=DEFAULT_TEXT_COLUMN_QUZ,
        text_column_spa=DEFAULT_TEXT_COLUMN_SPA,
        source_column="source",  # not present in this dataset; eval_mt.py handles that gracefully
        model=model,
        revision=revision,
        max_length=max_length,
        batch_size=batch_size,
        limit=limit,
        results_path=results_path,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            f"Evaluate a translation model against the small (61-row) hand-curated "
            f"medical-domain golden set ({DEFAULT_DATASET}), using the same BLEU/chrF++ "
            f"scoring as eval_mt.py. This measures out-of-domain generalization -- the "
            f"model's training/eval data (americasnlp2024/jw300) is religious-register "
            f"text, so a quality drop here relative to the main test-split scores is "
            f"expected, not a bug."
        )
    )
    parser.add_argument("--model", default=eval_mt.DEFAULT_MODEL, help="Translation model repo id or local path to evaluate")
    parser.add_argument(
        "--revision",
        default=None,
        help="Specific Hub revision of --model to load (tag/branch/commit sha), e.g. 'epoch-1' -- defaults to the latest commit on main",
    )
    parser.add_argument("--max-length", type=int, default=eval_mt.DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Cap eval rows per direction for a quick smoke test")
    parser.add_argument("--results-path", default=DEFAULT_RESULTS_PATH, help="JSONL file to append this run's scores to")
    return parser.parse_args()


def main():
    cli_args = parse_args()
    args = build_args(
        model=cli_args.model,
        revision=cli_args.revision,
        limit=cli_args.limit,
        results_path=cli_args.results_path,
        batch_size=cli_args.batch_size,
        max_length=cli_args.max_length,
    )
    eval_mt.run(args)


if __name__ == "__main__":
    main()
