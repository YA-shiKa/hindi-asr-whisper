import requests
import librosa
import soundfile as sf

def get_transcript(url):
    try:
        res = requests.get(url)
        print("URL:", url, "STATUS:", res.status_code)

        if res.status_code != 200:
            return ""

        data = res.json()
        return " ".join([seg["text"] for seg in data])

    except Exception as e:
        print("ERROR:", e)
        return ""

def download_audio(url, filename):
    r = requests.get(url)
    with open(filename, "wb") as f:
        f.write(r.content)

def preprocess_audio(path):
    audio, _ = librosa.load(path, sr=16000)
    sf.write(path, audio, 16000)