import os
import pandas as pd

from predict import generate_predictions, load_model
from error_analysis import add_wer_column, sample_errors, add_error_category
from evaluate import evaluate_model

LIMIT = 5              
EVAL_LIMIT = 20        
RUN_BASELINE = True

os.makedirs("outputs", exist_ok=True)


if RUN_BASELINE:
    print("\nRunning BASELINE model...\n")
    generate_predictions(
        "openai/whisper-small",
        "outputs/baseline.csv",
        limit=LIMIT
    )
    df_base = add_wer_column("outputs/baseline.csv")
else:
    print("\nSkipping BASELINE...\n")


print("\nRunning FINETUNED model...\n")
generate_predictions(
    "outputs/finetuned_model",
    "outputs/finetuned.csv",
    limit=LIMIT
)

df_ft = add_wer_column("outputs/finetuned.csv")


df_ft = add_error_category(df_ft)
sample_errors(df_ft)

df_ft.to_csv("outputs/final_with_wer.csv", index=False)


print("\nEvaluating on FLEURS dataset...\n")

model_base, proc_base = load_model("openai/whisper-small")
model_ft, proc_ft = load_model("outputs/finetuned_model")

baseline_wer = evaluate_model(model_base, proc_base, limit=EVAL_LIMIT)
finetuned_wer = evaluate_model(model_ft, proc_ft, limit=EVAL_LIMIT)

df_summary = pd.DataFrame({
    "Model": ["Baseline", "Finetuned"],
    "WER": [baseline_wer, finetuned_wer]
})

df_summary.to_csv("outputs/wer_summary.csv", index=False)

print("\nWER Summary:")
print(df_summary)


print("\n✅ Pipeline completed successfully!\n")