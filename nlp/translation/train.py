import argparse
import os

from datasets import concatenate_datasets, load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from common import LANG_CODES, pick_device

load_dotenv()

HF_USERNAME = os.environ.get("HF_USERNAME", "gmezan")
DEFAULT_DATASET = f"{HF_USERNAME}/quechua-spanish-parallel"
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_VALIDATION_SPLIT = "validation"
DEFAULT_SOURCE_PREFIXES = ["americasnlp"]
DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"
DEFAULT_OUTPUT_DIR = "./nllb-quy-spa-lora"
DEFAULT_HUB_REPO_ID = f"{HF_USERNAME}/nllb-quechua-spanish"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune NLLB-200 (LoRA) for Quechua<->Spanish translation, "
            "both directions in a single model, pulling the parallel dataset "
            "straight from the Hugging Face Hub. Trains on --train-split and "
            "validates on --validation-split -- native splits fixed at the "
            "data level (see preprocessing/scripts/assign_splits.py), shared "
            "with stt/ and tts/, so 'test' is never loaded here and stays a "
            "clean benchmark for evaluation/eval_mt.py."
        )
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="HF dataset repo id following preprocessing/docs/dataset-schema.md (text_quy/text_spa columns)",
    )
    parser.add_argument("--train-split", default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument("--validation-split", default=DEFAULT_VALIDATION_SPLIT)
    parser.add_argument("--text-column-quy", default="text_quy")
    parser.add_argument("--text-column-spa", default="text_spa")
    parser.add_argument("--source-column", default="source")
    parser.add_argument(
        "--source-prefixes",
        nargs="+",
        default=DEFAULT_SOURCE_PREFIXES,
        help=(
            "Only rows whose --source-column starts with one of these "
            "prefixes are used for training/validation -- an explicit "
            "allow-list rather than 'everything with non-blank text_spa', "
            "so a new source never gets pulled into training just by "
            "having both languages populated. Currently only "
            "'americasnlp' sources are cleaned (see "
            "preprocessing/scripts/clean_text.py) and dialect-verified for "
            "this pair."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base NLLB-200 checkpoint")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "v_proj"],
    )

    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)

    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help=(
            "Merge the LoRA adapter into the base model and push the final "
            "weights to the Hub (needs HF_TOKEN, e.g. via a .env file)"
        ),
    )
    parser.add_argument(
        "--hub-repo-id",
        default=DEFAULT_HUB_REPO_ID,
        help="Target model repo id. Defaults to '$HF_USERNAME/nllb-quechua-spanish'",
    )
    parser.add_argument("--private", action="store_true", help="Create the Hub repo as private")

    args = parser.parse_args()
    if args.push_to_hub and not args.hub_repo_id:
        parser.error("--hub-repo-id is required with --push-to-hub")
    return args


def build_bidirectional_dataset(dataset, text_col_quy, text_col_spa):
    """Turn each aligned (quy, spa) row into two training examples, one per
    translation direction, so a single fine-tune covers both quy->spa and
    spa->quy (NLLB is a single many-to-many model, not per-direction)."""
    quy_to_spa = dataset.map(
        lambda ex: {
            "src_text": ex[text_col_quy],
            "tgt_text": ex[text_col_spa],
            "src_lang": LANG_CODES["quy"],
            "tgt_lang": LANG_CODES["spa"],
        },
        remove_columns=dataset.column_names,
    )
    spa_to_quy = dataset.map(
        lambda ex: {
            "src_text": ex[text_col_spa],
            "tgt_text": ex[text_col_quy],
            "src_lang": LANG_CODES["spa"],
            "tgt_lang": LANG_CODES["quy"],
        },
        remove_columns=dataset.column_names,
    )
    combined = concatenate_datasets([quy_to_spa, spa_to_quy])
    return combined.filter(lambda ex: ex["src_text"] and ex["tgt_text"])


def main():
    args = parse_args()
    device = pick_device()
    print(f"Device: {device}")

    print(f"1. Loading parallel dataset ({args.dataset})...")
    columns = [args.text_column_quy, args.text_column_spa, args.source_column]

    def allowed_source(ex):
        return str(ex[args.source_column]).startswith(tuple(args.source_prefixes))

    train_raw = load_dataset(args.dataset, split=args.train_split).select_columns(columns).filter(allowed_source)
    validation_raw = load_dataset(args.dataset, split=args.validation_split).select_columns(columns).filter(allowed_source)
    print(f"   {len(train_raw)} train rows, {len(validation_raw)} validation rows (source startswith {args.source_prefixes})")

    train_dataset = build_bidirectional_dataset(train_raw, args.text_column_quy, args.text_column_spa)
    eval_dataset = build_bidirectional_dataset(validation_raw, args.text_column_quy, args.text_column_spa)

    print(f"2. Loading NLLB tokenizer and base model ({args.model})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.config.use_cache = False

    print("3. Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def preprocess(example):
        # Direction varies per row, so src_lang/tgt_lang must be set on the
        # tokenizer per-example (not once for the whole dataset) before encoding.
        tokenizer.src_lang = example["src_lang"]
        tokenizer.tgt_lang = example["tgt_lang"]
        return tokenizer(
            example["src_text"],
            text_target=example["tgt_text"],
            max_length=args.max_length,
            truncation=True,
        )

    print("4. Tokenizing...")
    train_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(preprocess, remove_columns=eval_dataset.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    print("5. Setting up training arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        seed=args.seed,
        predict_with_generate=True,
        fp16=(device == "cuda"),  # mps doesn't support fp16 training reliably; cpu doesn't support it at all
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("6. Training...")
    trainer.train()

    adapter_dir = f"{args.output_dir}-adapter"
    print(f"7. Saving LoRA adapter to {adapter_dir}...")
    model.save_pretrained(adapter_dir)

    if args.push_to_hub:
        print(f"8. Merging LoRA adapter into the base model and pushing to the Hub ({args.hub_repo_id})...")
        merged_model = model.merge_and_unload()
        merged_model.push_to_hub(args.hub_repo_id, private=args.private)
        tokenizer.push_to_hub(args.hub_repo_id, private=args.private)

    print("Done! Translation model is ready.")


if __name__ == "__main__":
    main()
