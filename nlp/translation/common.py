import torch

# NLLB-200 FLORES-200 language codes for this project's pair.
# quy_Latn = Ayacucho Quechua (closest NLLB-supported code to the Chanka
# variant this project otherwise uses; see nlp/README.md).
LANG_CODES = {
    "quy": "quy_Latn",
    "spa": "spa_Latn",
}


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
