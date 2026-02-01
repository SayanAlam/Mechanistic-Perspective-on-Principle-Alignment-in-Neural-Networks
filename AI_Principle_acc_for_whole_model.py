import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# Load the CSV
csv_url = "google sheet url"
df = pd.read_csv(csv_url)

print("Dataset loaded successfully!")
print(f"Total samples: {len(df)}\n")

# Load model
model_name = "Qwen/Qwen2.5-0.5B"
print(f"Loading model: {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

print("Model loaded successfully!\n")

def get_batch_predictions(questions, batch_size=16):
    prompts = []
    for question in questions:
        text = f"Question: {question}\nAnswer: "
        prompts.append(text)

    # Tokenize
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    id_A = tokenizer.encode("A", add_special_tokens=False)[0]
    id_B = tokenizer.encode("B", add_special_tokens=False)[0]

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[:, -1, :]

        scores_A = next_token_logits[:, id_A]
        scores_B = next_token_logits[:, id_B]

    responses = []
    for sA, sB in zip(scores_A, scores_B):
        responses.append("A" if sA > sB else "B")

    return responses

# ============ CONFIGURE BATCH SIZE HERE ============
BATCH_SIZE = 8
# ==================================================

predictions = []
questions_list = df['question'].tolist()
answers_list = df['answer'].tolist()
total_samples = len(df)

print("=" * 60)
print(f"Processing {total_samples} samples with batch size {BATCH_SIZE}...")
print("=" * 60)

for i in range(0, total_samples, BATCH_SIZE):
    batch_end = min(i + BATCH_SIZE, total_samples)
    batch_questions = questions_list[i:batch_end]
    batch_answers = answers_list[i:batch_end]

    batch_responses = get_batch_predictions(batch_questions, BATCH_SIZE)

    for j, (question, response, true_answer) in enumerate(zip(batch_questions, batch_responses, batch_answers)):
        true_answer = str(true_answer).strip().upper()

        predictions.append({
            'question': question,
            'true_answer': true_answer,
            'predicted_answer': response
        })

    if (batch_end % (BATCH_SIZE * 5)) == 0 or batch_end == total_samples:
        print(f"\nProcessed: {batch_end}/{total_samples} samples...")

# Calculate accuracy
correct = sum(1 for p in predictions if p['predicted_answer'] == p['true_answer'])
total = len(predictions)
accuracy = (correct / total) * 100 if total > 0 else 0

print("\n" + "=" * 60)
print("FINAL RESULTS:")
print("=" * 60)
print(f"Total Samples: {total}")
print(f"Correct Predictions: {correct}")
print(f"Incorrect Predictions: {total - correct}")
print(f"Accuracy: {accuracy:.2f}%")
print("=" * 60)

