def build_lattice(model_outputs):
    lattice = []
    max_len = max(len(m.split()) for m in model_outputs)

    for i in range(max_len):
        bin_words = set()

        for m in model_outputs:
            words = m.split()
            if i < len(words):
                bin_words.add(words[i])

        lattice.append(list(bin_words))

    return lattice


def lattice_wer(reference, lattice):
    ref_words = reference.split()
    score = 0

    for i, word in enumerate(ref_words):
        if i < len(lattice) and word in lattice[i]:
            score += 1

    return 1 - (score / len(ref_words))


def run_example():
    reference = "उसने चौदह किताबें खरीदीं"

    model_outputs = [
        "उसने 14 किताबें खरीदी",
        "उसने चौदह किताबे खरीदीं",
        "उसने चौदह पुस्तकें खरीदीं",
        "उसने चौदह किताबें खरीदीं"
    ]

    lattice = build_lattice(model_outputs)

    print("LATTICE:", lattice)
    print("Lattice WER:", lattice_wer(reference, lattice))


if __name__ == "__main__":
    run_example()