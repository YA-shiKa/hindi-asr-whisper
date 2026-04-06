from datasets import load_dataset
from jiwer import wer
from cleanup.number_normalizer import apply_all_fixes


def evaluate_model(model, processor, limit=20):  
    dataset = load_dataset(
        "google/fleurs",
        "hi_in",
        split="test",
        trust_remote_code=True
    )

    # 🔥 LIMIT DATA (IMPORTANT)
    if limit is not None:
        dataset = dataset.select(range(limit))

    refs, preds = [], []

    for i, sample in enumerate(dataset):
        print(f"Evaluating {i+1}/{len(dataset)}...")  

        audio = sample["audio"]["array"]

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

        pred_ids = model.generate(
            inputs["input_features"],
            forced_decoder_ids=processor.get_decoder_prompt_ids(
                language="hi",
                task="transcribe"
            )
        )

        pred = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        pred = apply_all_fixes(pred)

        refs.append(sample["transcription"])
        preds.append(pred)

    return wer(refs, preds)