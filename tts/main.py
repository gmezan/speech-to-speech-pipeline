import argparse
import csv
import io
import os

import librosa
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# Same underlying Quechua speech corpus `stt/` trains on, but pulled in its
# *raw* form (audio + transcript) rather than the Whisper-preprocessed form
# `preprocessing/` produces: VITS needs raw waveform + plain text, not
# log-Mel features tokenized against Whisper's vocabulary (see README).
DEFAULT_DATASET = "josemercado/quechua-chanka-asr"
DEFAULT_SPLIT = "train"
DEFAULT_AUDIO_COLUMN = "audio"
DEFAULT_TEXT_COLUMN = "TEXTO"
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_DATA_DIR = "./tts_data"
DEFAULT_OUTPUT_PATH = "./vits-stt-output"
DEFAULT_RUN_NAME = "quechua_vits"
SPEAKER_NAME = "quechua_speaker"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a VITS TTS model on the Quechua speech corpus."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset repo id (raw audio + text)")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--audio-column", default=DEFAULT_AUDIO_COLUMN)
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only materialize/train on the first N examples (quick smoke tests, e.g. on a Mac)",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Where to materialize the dataset as wavs/+metadata.csv (VITS' trainer reads from disk, not HF Dataset objects)",
    )
    parser.add_argument(
        "--skip-materialize",
        action="store_true",
        help="Reuse an already-materialized --data-dir instead of re-downloading/re-decoding audio",
    )

    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--eval-split-size", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-loader-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--save-step", type=int, default=1000)
    parser.add_argument("--print-step", type=int, default=25)
    parser.add_argument(
        "--restore-path",
        default="",
        help="Path or URL to an existing VITS checkpoint to continue from (transfer learning); default trains from scratch",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        default=torch.cuda.is_available(),
        help="Defaults to on when a CUDA GPU is available, off otherwise (unsupported on CPU)",
    )

    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the trained run (checkpoints + config) to the Hub (needs HF_TOKEN, e.g. via a .env file)",
    )
    parser.add_argument(
        "--hub-repo-id",
        help="Target model repo id, e.g. 'myorg/vits-quechua'. Required with --push-to-hub",
    )
    parser.add_argument("--private", action="store_true", help="Create the Hub repo as private")

    args = parser.parse_args()
    if args.push_to_hub and not args.hub_repo_id:
        parser.error("--hub-repo-id is required with --push-to-hub")
    return args


def materialize_dataset(dataset_name, split, audio_column, text_column, sample_rate, data_dir, limit):
    wavs_dir = os.path.join(data_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    print(f"Loading raw dataset ({dataset_name}, split={split})...")
    dataset = load_dataset(dataset_name, split=split)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    dataset = dataset.cast_column(audio_column, Audio(decode=False))

    print(f"Decoding {len(dataset)} clips, resampling to {sample_rate}Hz, writing wavs + metadata.csv...")
    texts = []
    metadata_path = os.path.join(data_dir, "metadata.csv")
    with open(metadata_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
        for i, row in enumerate(dataset):
            audio = row[audio_column]
            source = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
            array, sr = sf.read(source, dtype="float32")
            if array.ndim > 1:
                array = array.mean(axis=1)
            if sr != sample_rate:
                array = librosa.resample(array, orig_sr=sr, target_sr=sample_rate)

            utt_id = f"quy_{i:05d}"
            sf.write(os.path.join(wavs_dir, f"{utt_id}.wav"), array, sample_rate)

            text = row[text_column]
            writer.writerow([utt_id, text, text])
            texts.append(text)

    return texts


def build_characters_config(texts):
    letters = set()
    punctuations = set()
    for text in texts:
        for ch in text:
            if ch == " ":
                continue
            if ch.isalpha():
                letters.add(ch)
            else:
                punctuations.add(ch)
    return CharactersConfig(
        characters_class="TTS.tts.utils.text.characters.Graphemes",
        characters="".join(sorted(letters)),
        punctuations="".join(sorted(punctuations)) + " ",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
    )


def main():
    load_dotenv()
    args = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}")

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata.csv",
        path=args.data_dir,
    )

    if args.skip_materialize:
        print(f"Reusing already-materialized dataset at {args.data_dir}...")
        with open(os.path.join(args.data_dir, "metadata.csv"), encoding="utf-8") as f:
            texts = [line.split("|", 2)[1] for line in f if line.strip()]
    else:
        texts = materialize_dataset(
            args.dataset,
            args.split,
            args.audio_column,
            args.text_column,
            args.sample_rate,
            args.data_dir,
            args.limit,
        )

    characters_config = build_characters_config(texts)
    sample_sentences = [[t] for t in texts[:3]] or VitsConfig().test_sentences

    config = VitsConfig(
        run_name=args.run_name,
        output_path=args.output_path,
        audio=VitsAudioConfig(sample_rate=args.sample_rate),
        datasets=[dataset_config],
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_loader_workers=args.num_loader_workers,
        num_eval_loader_workers=args.num_loader_workers,
        epochs=args.epochs,
        save_step=args.save_step,
        print_step=args.print_step,
        eval_split_size=args.eval_split_size,
        use_phonemes=False,  # no reliable Quechua phonemizer backend; train on raw graphemes instead
        characters=characters_config,
        compute_input_seq_cache=True,
        print_eval=False,
        mixed_precision=args.mixed_precision,
        test_sentences=sample_sentences,
    )

    print("Building audio processor and tokenizer...")
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    print("Loading samples...")
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_size=config.eval_split_size,
    )
    print(f"{len(train_samples)} train samples, {len(eval_samples)} eval samples")

    model = Vits(config, ap, tokenizer, speaker_manager=None)

    trainer = Trainer(
        TrainerArgs(restore_path=args.restore_path),
        config,
        args.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        parse_command_line_args=False,
    )

    print("Training...")
    trainer.fit()

    if args.push_to_hub:
        print(f"Pushing run ({trainer.output_path}) to the Hub ({args.hub_repo_id})...")
        api = HfApi()
        api.create_repo(args.hub_repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=args.hub_repo_id,
            folder_path=str(trainer.output_path),
            repo_type="model",
            allow_patterns=["*.pth", "*.json", "*.txt"],
        )

    print("Done! TTS model is ready.")


if __name__ == "__main__":
    main()
