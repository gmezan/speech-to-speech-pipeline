import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
FINETUNE_DIR = os.path.join(HERE, "finetune_hf_vits")
CONVERT_SCRIPT = os.path.join(FINETUNE_DIR, "convert_original_discriminator_checkpoint.py")
TRAIN_SCRIPT = os.path.join(FINETUNE_DIR, "run_vits_finetuning.py")
MONOTONIC_ALIGN_DIR = os.path.join(FINETUNE_DIR, "monotonic_align")

load_dotenv()

HF_USERNAME = os.environ.get("HF_USERNAME", "gmezan")

# Same underlying Quechua speech corpus `stt/` trains on, pulled in its raw
# form (audio + transcript) — fine-tuning here starts from Meta's
# pretrained MMS-TTS checkpoint for this exact language variant
# (facebook/mms-tts-quy), via the vendored `finetune_hf_vits/` toolkit
# (github.com/ylacombe/finetune-hf-vits, MIT licensed). See README.
DEFAULT_LANGUAGE_CODE = "quy"
DEFAULT_DATASET = f"{HF_USERNAME}/quechua-spanish-parallel"
DEFAULT_AUDIO_COLUMN = "audio_quy"
DEFAULT_TEXT_COLUMN = "text_quy"
DEFAULT_SAMPLE_TEXT = "Achikyaqta parayta qallarimun."
DEFAULT_OUTPUT_DIR = "./vits-quy-finetuned"
DEFAULT_HUB_REPO_ID = f"{HF_USERNAME}/vits-quechua"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune MMS-TTS (VITS) for Quechua on the speech corpus, via the vendored finetune_hf_vits toolkit."
    )
    parser.add_argument(
        "--language-code",
        default=DEFAULT_LANGUAGE_CODE,
        help="ISO 639-3 code of the MMS-TTS base checkpoint to fine-tune (facebook/mms-tts-<code>)",
    )
    parser.add_argument(
        "--base-checkpoint",
        default=None,
        help=(
            "Path or Hub repo id of an already discriminator-equipped base checkpoint. "
            "If unset, one is built once from --language-code and cached under "
            "finetune_hf_vits/mms-tts-<language-code>-train/"
        ),
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset repo id (raw audio + text)")
    parser.add_argument("--audio-column", default=DEFAULT_AUDIO_COLUMN)
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument(
        "--sample-text",
        default=DEFAULT_SAMPLE_TEXT,
        help="Text synthesized as an audio preview during eval",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap training samples for a quick smoke test (sets max_train_samples)",
    )
    parser.add_argument("--eval-samples", type=int, default=25, help="Number of held-out eval samples")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Defaults to on when a CUDA GPU is available, off otherwise",
    )

    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the finetuned model to the Hub (needs HF_TOKEN, e.g. via a .env file)",
    )
    parser.add_argument(
        "--hub-repo-id",
        default=DEFAULT_HUB_REPO_ID,
        help="Target model repo id. Defaults to '$HF_USERNAME/vits-quechua'",
    )
    parser.add_argument("--private", action="store_true", help="Create the Hub repo as private")

    args = parser.parse_args()
    if args.push_to_hub and not args.hub_repo_id:
        parser.error("--hub-repo-id is required with --push-to-hub")
    return args


def ensure_monotonic_align_built():
    if glob.glob(os.path.join(MONOTONIC_ALIGN_DIR, "*.so")):
        return
    print("Building the monotonic_align Cython extension (one-time)...")
    os.makedirs(os.path.join(MONOTONIC_ALIGN_DIR, "monotonic_align"), exist_ok=True)
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=MONOTONIC_ALIGN_DIR,
        check=True,
    )


def ensure_base_checkpoint(language_code):
    checkpoint_dir = os.path.join(FINETUNE_DIR, f"mms-tts-{language_code}-train")
    if os.path.isdir(checkpoint_dir):
        return checkpoint_dir
    print(f"Building the discriminator-equipped base checkpoint for '{language_code}' (one-time)...")
    subprocess.run(
        [
            sys.executable,
            CONVERT_SCRIPT,
            "--language_code",
            language_code,
            "--pytorch_dump_folder_path",
            checkpoint_dir,
        ],
        check=True,
    )
    return checkpoint_dir


def main():
    args = parse_args()

    ensure_monotonic_align_built()
    base_checkpoint = args.base_checkpoint or ensure_base_checkpoint(args.language_code)

    config = {
        "project_name": f"{args.language_code}_vits_finetuning",
        "output_dir": args.output_dir,
        "overwrite_output_dir": True,
        "report_to": [],
        "dataset_name": args.dataset,
        "audio_column_name": args.audio_column,
        "text_column_name": args.text_column,
        # The parallel dataset only has a "train" split; reuse it for eval too.
        "train_split_name": "train",
        "eval_split_name": "train",
        "full_generation_sample_text": args.sample_text,
        "max_duration_in_seconds": 20,
        "min_duration_in_seconds": 1.0,
        "max_tokens_length": 500,
        "model_name_or_path": base_checkpoint,
        "preprocessing_num_workers": 4,
        "do_train": True,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "adam_beta1": 0.8,
        "adam_beta2": 0.99,
        "do_eval": True,
        "eval_steps": args.eval_steps,
        "per_device_eval_batch_size": args.eval_batch_size,
        "max_eval_samples": args.eval_samples,
        "do_step_schedule_per_epoch": True,
        "weight_disc": 3,
        "weight_fmaps": 1,
        "weight_gen": 1,
        "weight_kl": 1.5,
        "weight_duration": 1,
        "weight_mel": 35,
        "fp16": args.fp16,
        "seed": 456,
    }
    if config["fp16"] is None:
        import torch

        config["fp16"] = torch.cuda.is_available()
    if args.limit:
        config["max_train_samples"] = args.limit
    if args.push_to_hub:
        config["push_to_hub"] = True
        config["hub_model_id"] = args.hub_repo_id
        config["hub_private_repo"] = args.private

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(config, f, indent=2)
        config_path = f.name

    print(f"Training config written to {config_path}:")
    print(json.dumps(config, indent=2))

    subprocess.run([sys.executable, TRAIN_SCRIPT, config_path], check=True)


if __name__ == "__main__":
    main()
