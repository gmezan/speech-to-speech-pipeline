import argparse
import json
import os
import re

from dotenv import load_dotenv
from huggingface_hub import ModelCard

load_dotenv()

HF_USERNAME = os.environ.get("HF_USERNAME", "gmezan")
DEFAULT_MODEL = f"{HF_USERNAME}/nllb-quechua-spanish"
SECTION_HEADING = "## Evaluation results"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Publish an eval_mt.py run's full per-direction/per-source BLEU/chrF++ breakdown "
            "onto --model's Hub model card (README.md), under a block scoped to --label -- so "
            "every tagged epoch documents its own eval results in one place on the model card, "
            "without clobbering other epochs' blocks. Re-running for the same --label replaces "
            "that block in place rather than duplicating it."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hub model repo id to update the card of")
    parser.add_argument(
        "--label",
        required=True,
        help="Section label for this eval run, e.g. 'epoch-2' -- normally matches the --tag used when pushing the model (nlp/translation/train.py --tag)",
    )
    parser.add_argument(
        "--results-path",
        required=True,
        help="JSONL file produced by evaluation/eval_mt.py -- its last line is published",
    )
    return parser.parse_args()


def build_block(label, results):
    lines = [f"<!-- eval:{label} -->", f"### `{label}`", ""]
    lines.append(f"- **Dataset**: `{results['dataset']}` (split=`{results['split']}`)")
    if results.get("revision") and results["revision"] != label:
        lines.append(f"- **Note**: eval_mt.py was run with `--revision {results['revision']}`, which differs from this block's label (`{label}`)")
    if results.get("sentences_per_second"):
        lines.append(f"- **Throughput**: {results['sentences_per_second']} sentences/sec")
    if results.get("timestamp"):
        lines.append(f"- **Run date**: {results['timestamp']}")
    lines.append("")
    lines.append("| direction | source | n | BLEU | chrF++ |")
    lines.append("|---|---|---:|---:|---:|")
    for direction, by_source in results["directions"].items():
        for source, scores in sorted(by_source.items(), key=lambda kv: kv[0] != "__all__"):
            source_label = "ALL" if source == "__all__" else source
            lines.append(f"| `{direction}` | {source_label} | {scores['n']} | {scores['bleu']:.2f} | {scores['chrf++']:.2f} |")
    lines.append(f"<!-- /eval:{label} -->")
    return "\n".join(lines)


def upsert_section(card_text, label, block):
    """Replace this label's block in place if it already exists (so re-running the same
    label updates rather than duplicates); otherwise append it under SECTION_HEADING,
    creating that heading on first use. Other labels' blocks are left untouched either way,
    so every epoch's results accumulate on the same card instead of overwriting each other."""
    marker_re = re.compile(rf"<!-- eval:{re.escape(label)} -->.*?<!-- /eval:{re.escape(label)} -->", re.DOTALL)
    if marker_re.search(card_text):
        return marker_re.sub(block, card_text)

    if SECTION_HEADING not in card_text:
        card_text = card_text.rstrip() + f"\n\n{SECTION_HEADING}\n\n" + block + "\n"
    else:
        card_text = card_text.rstrip() + "\n\n" + block + "\n"
    return card_text


def main():
    args = parse_args()

    print(f"1. Loading results from {args.results_path}...")
    with open(args.results_path) as f:
        lines = f.readlines()
    if not lines:
        raise ValueError(f"{args.results_path} is empty -- run evaluation/eval_mt.py first")
    results = json.loads(lines[-1])

    print(f"2. Loading model card for {args.model}...")
    card = ModelCard.load(args.model)

    print(f"3. Updating '{args.label}' section...")
    block = build_block(args.label, results)
    card.text = upsert_section(card.text, args.label, block)

    print(f"4. Pushing updated card to {args.model}...")
    card.push_to_hub(args.model, commit_message=f"Document {args.label} eval results")
    print("Done!")


if __name__ == "__main__":
    main()
