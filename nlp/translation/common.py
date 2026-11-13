import torch

# NLLB-200 FLORES-200 language codes for this project's pair.
#
# "quz_Latn" is NOT a real pretrained NLLB language code (verified against
# facebook/nllb-200-distilled-600M's tokenizer: convert_tokens_to_ids("quz_Latn")
# returns the UNK token id, not a dedicated language embedding) -- an earlier
# version of this project used it anyway, meaning every "quz"-tagged training
# example was actually tagged with UNK. "quy_Latn" (Ayacucho/Chanka Quechua) is
# the real pretrained code NLLB ships, and the closest one available -- using it
# lets fine-tuning build on NLLB's actual Quechua knowledge instead of learning
# the language association from scratch via an overloaded UNK token. The dict
# key stays "quz" (matching this project's column names/dataset, which really
# is Collao Quechua); only the NLLB-facing code value changed.
LANG_CODES = {
    "quz": "quy_Latn",
    "spa": "spa_Latn",
}


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
