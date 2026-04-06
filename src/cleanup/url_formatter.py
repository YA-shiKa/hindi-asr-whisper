def fix_urls(original_audio_url):
    """
    Convert old joshtalks URL → new upload_goai URL
    """

    parts = original_audio_url.split("/")

    user_folder = parts[-2]        
    file_name = parts[-1]          

    recording_id = file_name.split("_")[0]

    base = "https://storage.googleapis.com/upload_goai"

    transcription = f"{base}/{user_folder}/{recording_id}_transcription.json"
    audio = f"{base}/{user_folder}/{recording_id}_audio.wav"

    return transcription, audio