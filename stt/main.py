import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

load_dotenv()

HF_USERNAME = os.environ.get("HF_USERNAME", "gmezan")
DEFAULT_DATASET = f"{HF_USERNAME}/quechua-whisper-ready"
DEFAULT_SPLIT = "train"
DEFAULT_EVAL_SIZE = 0.1
DEFAULT_MODEL = "openai/whisper-small"
DEFAULT_OUTPUT_DIR = "./whisper-stt-lora"
DEFAULT_HUB_REPO_ID = f"{HF_USERNAME}/whisper-quechua"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune Whisper (LoRA) on a training-ready ASR dataset pulled "
            "straight from the Hugging Face Hub."
        )
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="HF dataset repo id produced by preprocessing/main.py (input_features/labels columns)",
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--eval-size",
        type=float,
        default=DEFAULT_EVAL_SIZE,
        help="Fraction of --split held out for eval; 0 reuses the train set as eval",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base Whisper checkpoint")
    parser.add_argument(
        "--language",
        default="spanish",
        help="Must match the language the processor was built with in preprocessing",
    )
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=["q_proj", "v_proj"])

    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--generation-max-length", type=int, default=225)

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
        help="Target model repo id. Defaults to '$HF_USERNAME/whisper-quechua'",
    )
    parser.add_argument("--private", action="store_true", help="Create the Hub repo as private")

    args = parser.parse_args()
    if args.push_to_hub and not args.hub_repo_id:
        parser.error("--hub-repo-id is required with --push-to-hub")
    return args


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 so it's ignored in the loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main():
    args = parse_args()
    device = pick_device()
    print(f"Device: {device}")

    print(f"1. Loading training-ready dataset ({args.dataset}, split={args.split})...")
    dataset = load_dataset(args.dataset, split=args.split)
    if args.eval_size > 0:
        split_dataset = dataset.train_test_split(test_size=args.eval_size, seed=args.seed)
        train_dataset, eval_dataset = split_dataset["train"], split_dataset["test"]
    else:
        train_dataset, eval_dataset = dataset, dataset

    print(f"2. Loading Whisper processor and base model ({args.model})...")
    processor = WhisperProcessor.from_pretrained(args.model, language=args.language, task=args.task)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
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

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    print("4. Setting up training arguments...")
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
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        remove_unused_columns=False,  # dataset columns are input_features/labels, not model kwarg names
        label_names=["labels"],
        fp16=(device == "cuda"),  # mps doesn't support fp16 training reliably; cpu doesn't support it at all
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
    )

    print("5. Training...")
    trainer.train()

    adapter_dir = f"{args.output_dir}-adapter"
    print(f"6. Saving LoRA adapter to {adapter_dir}...")
    model.save_pretrained(adapter_dir)

    if args.push_to_hub:
        print(f"7. Merging LoRA adapter into the base model and pushing to the Hub ({args.hub_repo_id})...")
        merged_model = model.merge_and_unload()
        merged_model.push_to_hub(args.hub_repo_id, private=args.private)
        processor.push_to_hub(args.hub_repo_id, private=args.private)

    print("Done! STT model is ready.")


if __name__ == "__main__":
    main()
