import jiwer
import torch

# Both error rates ignore casing/punctuation/whitespace differences that don't
# reflect real transcription/synthesis errors (e.g. Whisper's punctuation
# style vs. the corpus's), so scores track content accuracy, not formatting.
_NORMALIZE = [
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
]
WER_TRANSFORM = jiwer.Compose([*_NORMALIZE, jiwer.ReduceToListOfListOfWords()])
CER_TRANSFORM = jiwer.Compose([*_NORMALIZE, jiwer.ReduceToListOfListOfChars()])


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_error_rates(references, hypotheses):
    wer = jiwer.wer(
        references,
        hypotheses,
        reference_transform=WER_TRANSFORM,
        hypothesis_transform=WER_TRANSFORM,
    )
    cer = jiwer.cer(
        references,
        hypotheses,
        reference_transform=CER_TRANSFORM,
        hypothesis_transform=CER_TRANSFORM,
    )
    return wer, cer
