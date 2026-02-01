import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load GPT-2 Medium with "eager" attention
# We must use attn_implementation="eager" to be able to access attention weights
print("Loading GPT-2 Medium...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = GPT2LMHeadModel.from_pretrained(
    "gpt2-medium",
    attn_implementation="eager"
).to(device)

model.eval()
model.config.output_attentions = True

sheet_url = "https://docs.google.com/spreadsheets/d/1THZHNOj7aAFbS6SPJoLKnnm86PY0hO2ctCc15VnMJ7k/export?format=csv"

try:
    df = pd.read_csv(sheet_url)

    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Check if 'question' column exists (case-insensitive check)
    # This finds the column even if it's named "Question" or "question "
    col_map = {c.lower(): c for c in df.columns}
    if 'question' not in col_map:
        raise ValueError(f"Column 'question' not found. Available columns: {list(df.columns)}")

    target_col = col_map['question']

except Exception as e:
    print(f"Error loading CSV: {e}")
    # Stop execution if data load fails
    df = pd.DataFrame()

df["text_length"] = df["question"].str.len()
subset = df.nsmallest(10, "text_length")

print(f"Successfully loaded {len(df)} rows. Processing the first {len(subset)}...\n")

for i, row in subset.iterrows():
    gpt2_text = str(row[target_col])

    # Skip empty input
    if not gpt2_text.strip() or gpt2_text.lower() == 'nan':
        print(f"Skipping Row {i+1}: Input is empty.")
        continue

    # Prepare Data
    inputs = tokenizer(gpt2_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    tokens = [tokenizer.decode([t]).replace('Ġ', ' ') for t in input_ids[0]]

    # Run Model
    with torch.no_grad():
        outputs = model(**inputs)

    # Process Attention
    if outputs.attentions is None:
        print("Error: Attentions not found.")
    else:
        # Select Layer (0 to 23 for GPT-2 Medium)
        layer_idx = 8

        # Get attention: (batch, num_heads, seq_len, seq_len) -> Take batch 0
        layer_attention = outputs.attentions[layer_idx][0]

        # Average across heads (dim=0)
        avg_attention = layer_attention.mean(dim=0)
        print(f"Computed average attention for Layer {layer_idx}. Shape: {avg_attention.shape}")

        # Visualize
        plt.figure(figsize=(7, 7))
        sns.heatmap(
            avg_attention.cpu().numpy(),
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="Reds",
            square=True,
            cbar_kws={'label': 'Attention Weight'}
        )
        plt.title(f"GPT-2 Medium: Layer {layer_idx} Average Attention", fontsize=14)
        plt.xticks(rotation=90, fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        plt.show()



        last_token_attn = avg_attention[-1, :]

        top_values, top_indices = torch.topk(last_token_attn, 10)

        current_token_str = tokens[-1]
        print(f"The last token processed was: '{current_token_str}'")
        print(f"Here are the top 10 tokens it paid attention to (Layer {layer_idx}):\n")

        print(f"{'Rank':<5} | {'Token':<20} | {'Attention Score':<15}")
        print("-" * 50)

        for i in range(10):
            idx = top_indices[i].item()
            score = top_values[i].item()
            token_str = tokens[idx]

            # Clean up newlines for cleaner printing
            display_token = token_str.replace('\n', '\\n')

            print(f"{i+1:<5} | {display_token:<20} | {score:.4f}")

