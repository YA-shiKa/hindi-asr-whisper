# Hindi ASR Fine-Tuning & Error Analysis (Whisper)

---

## Overview

This project focuses on building a Hindi Automatic Speech Recognition (ASR) system by fine-tuning OpenAI’s Whisper-small model using PyTorch and Hugging Face Transformers.

It implements a complete end-to-end ML pipeline:

- Data preprocessing
- Model fine-tuning
- Inference & evaluation (WER)
- Error analysis & taxonomy
- Post-processing (text cleanup)

---

## Key Features
- Fine-tuned Whisper-small for Hindi speech recognition
- Full data pipeline (audio + transcription processing)
- Evaluation using Word Error Rate (WER)
- Error analysis system with categorization
- Text cleanup pipeline:
  -Number normalization (Hindi → digits)
  -English word detection in Hindi text
- Lattice-based evaluation for flexible ASR scoring
- Modular and scalable code structure

---

## Tech Stack
- PyTorch
- Hugging Face Transformers
- Whisper (ASR)
- Librosa (audio processing)
- JiWER (evaluation)
- Pandas 
