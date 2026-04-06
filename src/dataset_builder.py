from preprocess import get_transcript
from url_formatter import fix_urls

def build_dataset(df):
    data = []

    for _, row in df.iterrows():
        transcription_url, audio_url = fix_urls(row["rec_url_gcp"])

        text = get_transcript(transcription_url)

        print("TEXT SAMPLE:", text[:50]) 

        if text.strip() == "":
            continue

        data.append({
            "audio_url": audio_url,
            "text": text,
            "duration": row["duration"]
        })

    return data