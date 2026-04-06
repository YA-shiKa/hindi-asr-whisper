import torch
import librosa
import pandas as pd
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from cleanup.number_normalizer import apply_all_fixes
from cleanup.english_detector import detect_english


def load_model(path):
    processor = WhisperProcessor.from_pretrained(path)
    model = WhisperForConditionalGeneration.from_pretrained(path)
    model.eval()
    return model, processor


def predict(model, processor, audio_path):
    audio, _ = librosa.load(audio_path, sr=16000)

    chunk_size = 15 * 16000  # 15 sec chunks
    chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]

    full_text = []

    for chunk in chunks:
        inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            pred_ids = model.generate(
                inputs["input_features"],
                forced_decoder_ids=processor.get_decoder_prompt_ids(
                    language="hi",
                    task="transcribe"
                ),
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                length_penalty=1.0,
                early_stopping=True
            )

        pred = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        full_text.append(pred)

    return " ".join(full_text)


def generate_predictions(model_path, save_path, limit=None):
    import pandas as pd

    df = pd.read_csv("data/processed/final_dataset.csv")

    if limit is not None:
        df = df.head(limit)

    model, processor = load_model(model_path)

    preds = []

    for i, row in df.iterrows():
        print(f"Processing {i}...")
        pred = predict(model, processor, row["audio_path"])
        preds.append(pred)

    df["prediction"] = preds

    df["prediction"] = df["prediction"].apply(apply_all_fixes)
    df["prediction"] = df["prediction"].apply(detect_english)

    df.to_csv(save_path, index=False)
    print(f"Saved → {save_path}")