import re

NUM_MAP = {
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5,
    "छह": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10
}

TENS = {
    "बीस": 20, "तीस": 30, "चालीस": 40,
    "पचास": 50, "साठ": 60, "सत्तर": 70,
    "अस्सी": 80, "नब्बे": 90
}

MULTIPLIERS = {
    "सौ": 100,
    "हज़ार": 1000
}


def is_idiom(text):
    return "दो चार" in text or "एक दिन" in text


def parse_number_phrase(words):
    total = 0
    current = 0

    for w in words:
        if w in NUM_MAP:
            current += NUM_MAP[w]
        elif w in TENS:
            current += TENS[w]
        elif w in MULTIPLIERS:
            current *= MULTIPLIERS[w]
            total += current
            current = 0

    return str(total + current) if (total + current) > 0 else None


def normalize_numbers(text):
    if is_idiom(text):
        return text

    words = text.split()
    result = []
    buffer = []

    for w in words:
        if w in NUM_MAP or w in TENS or w in MULTIPLIERS:
            buffer.append(w)
        else:
            if buffer:
                num = parse_number_phrase(buffer)
                result.append(num if num else " ".join(buffer))
                buffer = []
            result.append(w)

    if buffer:
        num = parse_number_phrase(buffer)
        result.append(num if num else " ".join(buffer))

    return " ".join(result)


def remove_repetition(text):
    words = text.split()
    cleaned = []

    for w in words:
        if not cleaned or cleaned[-1] != w:
            cleaned.append(w)

    text = " ".join(cleaned)
    text = re.sub(r'(.)\1{3,}', r'\1', text)

    return text


def apply_all_fixes(text):
    text = remove_repetition(text)
    text = normalize_numbers(text)
    return text