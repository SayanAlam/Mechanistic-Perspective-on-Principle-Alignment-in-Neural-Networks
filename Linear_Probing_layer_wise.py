import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
from sklearn.model_selection import train_test_split
import random
import os

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

set_seed(42)

gc.collect()
torch.cuda.empty_cache()

model_name = "meta-llama/Llama-3.2-1B"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    output_hidden_states=True,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model.eval()

# Qwen tokenizer setup
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

url = "google sheet url"
df = pd.read_csv(url, engine="python")

texts = []
labels = []

print("Processing Data...")
for i in range(len(df)):
    inp = df["question"][i]
    ans = df["answer"][i]

    if pd.isna(ans) or pd.isna(inp):
        continue

    ans = str(ans).strip()
    if ans not in ["A", "B"]:
        continue

    texts.append(str(inp))
    labels.append(0 if ans == "A" else 1)

print(f"Total valid samples: {len(labels)}")
labels = torch.tensor(labels, dtype=torch.long)

batch_size = 16
text_loader = DataLoader(texts, batch_size=batch_size, shuffle=False)

# Dictionary to store embeddings for EVERY layer
# Keys will be layer indices (0 to N), Values will be list of batch tensors
layer_data = {}

print("Extracting Features from ALL layers...")

with torch.no_grad():
    for batch_texts in tqdm(text_loader):
        inputs = tokenizer(
            list(batch_texts),
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        outputs = model(**inputs)

        # outputs.hidden_states is a tuple of (num_layers + 1) tensors
        # Index 0 is embedding layer, Index 1 is first block, etc.
        all_states = outputs.hidden_states

        batch_size_curr = all_states[0].shape[0]
        # Find index of last token for each sequence
        sequence_lengths = inputs.attention_mask.sum(dim=1) - 1

        for layer_idx, layer_tensor in enumerate(all_states):
            if layer_idx not in layer_data:
                layer_data[layer_idx] = []

            # Extract last token vector: [Batch_Size, Hidden_Dim]
            batch_embeddings = layer_tensor[torch.arange(batch_size_curr, device=device), sequence_lengths]

            # Move to CPU to save GPU memory for the next batch
            layer_data[layer_idx].append(batch_embeddings.detach().cpu().to(torch.float32))

print("Feature extraction complete.")

# Clean up model to free VRAM for probing
del model
del inputs
del outputs
torch.cuda.empty_cache()

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.linear(x)

def train_probe_for_layer(layer_idx, embeddings_list, labels, num_runs=5):
    # Merge batches into one big tensor
    X = torch.cat(embeddings_list, dim=0).numpy()
    Y = labels.numpy()

    run_accuracies = []

    # Loop for the number of runs (5 times)
    for run in range(num_runs):
        # Generate a unique seed for this run
        current_seed = 42 + run

        # STRATIFIED SPLIT
        # We use 'current_seed' so every run has a different Train/Test split
        train_x_np, test_x_np, train_y_np, test_y_np = train_test_split(
            X, Y,
            test_size=0.2,
            random_state=current_seed,
            stratify=Y
        )

        # Convert back to Torch Tensors
        train_x = torch.tensor(train_x_np, dtype=torch.float32)
        test_x = torch.tensor(test_x_np, dtype=torch.float32)
        train_y = torch.tensor(train_y_np, dtype=torch.long)
        test_y = torch.tensor(test_y_np, dtype=torch.long)

        # Create Loaders
        train_ds = TensorDataset(train_x, train_y)
        test_ds = TensorDataset(test_x, test_y)
        train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

        # Init Probe
        # We re-initialize the model every loop to reset weights
        input_dim = X.shape[1]
        probe = LinearProbe(input_dim, 2).to(device)
        optimizer = optim.Adam(probe.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Train Loop
        epochs = 50
        probe.train()
        for epoch in range(epochs):
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = probe(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        # Eval Loop
        probe.eval()
        total_acc = 0
        total_count = 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = probe(xb)
                preds = torch.argmax(logits, dim=1)
                total_acc += (preds == yb).sum().item()
                total_count += yb.size(0)

        # Calculate accuracy for this single run
        acc = total_acc / total_count
        run_accuracies.append(acc)

    # Calculate Mean Accuracy across 5 runs
    mean_acc = sum(run_accuracies) / len(run_accuracies)
    std_acc = np.std(run_accuracies) # Optional: Standard deviation to see variance

    return mean_acc, std_acc # Return both mean and std

results = {}
print(f"\n{'='*40}")
print(f"Training Probes on {len(layer_data)} Layers (5 Runs Average)")
print(f"{'='*40}")

for layer_idx in sorted(layer_data.keys()):
    # Call with num_runs=5
    avg_acc, std_dev = train_probe_for_layer(layer_idx, layer_data[layer_idx], labels, num_runs=5)
    results[layer_idx] = avg_acc

    layer_name = "Embeddings" if layer_idx == 0 else f"Layer {layer_idx}"
    print(f"{layer_name:12} | Avg Acc: {avg_acc*100:.2f} (+/- {std_dev:.4f})")

best_layer = max(results, key=results.get)
best_acc = results[best_layer]

worst_layer = min(results, key=results.get)
worst_acc = results[worst_layer]

print(f"\n{'='*40}")
print(f" BEST LAYER : {best_layer} (Avg Acc: {best_acc:.4f})")
print(f" WORST LAYER: {worst_layer} (Avg Acc: {worst_acc:.4f})")
print(f"{'='*40}")