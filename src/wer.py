from jiwer import wer

def compute_wer(ref, pred):
    return wer(ref, pred)