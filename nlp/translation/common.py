import torch

# NLLB-200 FLORES-200 language codes for this project's pair.
# quz_Latn = Cusco-Collao Quechua, an exact match for this project's target
# variety (see nlp/README.md).
LANG_CODES = {
    "quz": "quz_Latn",
    "spa": "spa_Latn",
}


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
