import re

DEV_ENGLISH = ["कंप्यूटर", "इंटरव्यू", "जॉब", "प्रॉब्लम"]


def detect_english(text):
    words = text.split()
    tagged = []

    for w in words:
        if re.match(r'[a-zA-Z]+', w):
            tagged.append(f"[EN]{w}[/EN]")
        elif w in DEV_ENGLISH:
            tagged.append(f"[EN]{w}[/EN]")
        else:
            tagged.append(w)

    return " ".join(tagged)