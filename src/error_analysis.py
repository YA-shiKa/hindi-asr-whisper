import pandas as pd
from wer import compute_wer

def add_wer_column(path):
    df = pd.read_csv(path)

    df["wer"] = [
        compute_wer(r["text"], r["prediction"])
        for _, r in df.iterrows()
    ]

    df.to_csv(path, index=False)
    return df

def sample_errors(df, n=25):
    df = df[df["wer"] > 0].sort_values("wer", ascending=False)
    step = max(1, len(df)//n)
    sample = df.iloc[::step].head(n)

    sample.to_csv("outputs/error_samples.csv", index=False)
    return sample

def categorize_error(ref, pred):
    if ref == pred:
        return "correct"

    if any(c.isdigit() for c in pred):
        return "number_error"

    if "[EN]" in pred:
        return "code_mixing"

    if len(pred.split()) < len(ref.split()):
        return "deletion"

    if len(pred.split()) > len(ref.split()):
        return "insertion"

    return "substitution"

def add_error_category(df):
    df["error_type"] = [
        categorize_error(r["text"], r["prediction"])
        for _, r in df.iterrows()
    ]
    return df
