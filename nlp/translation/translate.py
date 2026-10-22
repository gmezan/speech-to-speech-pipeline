import argparse
import os

from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from common import LANG_CODES, pick_device

load_dotenv()

HF_USERNAME = os.environ.get("HF_USERNAME", "gmezan")
DEFAULT_MODEL = f"{HF_USERNAME}/nllb-quechua-spanish"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate a sentence with a fine-tuned (or base) NLLB-200 model."
    )
    parser.add_argument("text", help="Text to translate")
    parser.add_argument(
        "--src",
        choices=list(LANG_CODES),
        required=True,
        help="Source language",
    )
    parser.add_argument(
        "--tgt",
        choices=list(LANG_CODES),
        required=True,
        help="Target language",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model repo id. Defaults to '$HF_USERNAME/nllb-quechua-spanish'; "
        "pass 'facebook/nllb-200-distilled-600M' to try the untuned base model",
    )
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    if args.src == args.tgt:
        parser.error("--src and --tgt must differ")
    return args


def main():
    args = parse_args()
    device = pick_device()

    tokenizer = AutoTokenizer.from_pretrained(args.model, src_lang=LANG_CODES[args.src])
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)

    inputs = tokenizer(args.text, return_tensors="pt").to(device)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(LANG_CODES[args.tgt])
    generated = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=args.max_length,
    )
    print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
