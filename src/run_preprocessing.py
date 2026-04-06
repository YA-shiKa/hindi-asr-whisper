import os
import pandas as pd
from data_loader import load_csv
from dataset_builder import build_dataset
from preprocess import download_audio, preprocess_audio

def main():
    df = load_csv("data/raw_csv/dataasr.csv")

    data = build_dataset(df)
    clean_df = pd.DataFrame(data)

    os.makedirs("data/processed/audio", exist_ok=True)

    audio_paths = []

    for i, row in clean_df.iterrows():
        path = f"data/processed/audio/{i}.wav"
        download_audio(row["audio_url"], path)
        preprocess_audio(path)
        audio_paths.append(path)

    clean_df["audio_path"] = audio_paths

    clean_df = clean_df[
        (clean_df["duration"] > 1) &
        (clean_df["text"].str.len() > 5)
    ]

    clean_df.to_csv("data/processed/final_dataset.csv", index=False)

if __name__ == "__main__":
    main()