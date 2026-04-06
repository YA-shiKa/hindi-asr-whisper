import pandas as pd
import librosa
import torch
from datasets import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

MODEL_NAME = "openai/whisper-small"


def load_dataset():
    df = pd.read_csv("data/processed/final_dataset.csv")
    return Dataset.from_pandas(df[["audio_path", "text"]])


def prepare_dataset(dataset, processor):

    def preprocess(example):
        audio, _ = librosa.load(example["audio_path"], sr=16000)

        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        labels = processor.tokenizer(
            example["text"],
            return_tensors="pt",
            padding="max_length",
            max_length=256,
            truncation=True
        )

        return {
            "input_features": inputs.input_features.squeeze(0),
            "labels": labels.input_ids.squeeze(0)
        }

    dataset = dataset.map(preprocess)

    dataset = dataset.remove_columns(["audio_path", "text"])

    return dataset

class DataCollatorSpeechSeq2Seq:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [torch.tensor(f["input_features"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id
        )

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        input_features = torch.stack(input_features)

        return {
            "input_features": input_features,
            "labels": labels
        }


def train():
    dataset = load_dataset()

    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="hi",
        task="transcribe"
    )
    model.config.suppress_tokens = []

    dataset = prepare_dataset(dataset, processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir="outputs/finetuned_model",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        num_train_epochs=3,
        logging_dir="outputs/logs",
        save_steps=200,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        logging_steps=10
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor.feature_extractor,
        data_collator=DataCollatorSpeechSeq2Seq(processor),
    )

    trainer.train()

    model.save_pretrained("outputs/finetuned_model")
    processor.save_pretrained("outputs/finetuned_model")


if __name__ == "__main__":
    train()