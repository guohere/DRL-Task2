import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

"""
Use a test dataset (from the same source as training, e.g., trl-lib/ultrafeedback_binarized).
Check how often the reward model assigns a higher score to the preferred response.
Calculate Accuracy (higher accuracy = better alignment with human feedback).

🔹 Interpretation:
 If accuracy > 80%, the reward model is aligned with human preferences.
 If accuracy < 60%, the reward model may not generalize well.
"""
# ✅ Load Reward Model & Tokenizer

reward_model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen2.5-0.5B-Reward/checkpoint-182",
    num_labels=1  # ✅ Ensure correct output shape
)

tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-0.5B-Reward/checkpoint-182")

# ✅ Load Test Dataset (Separate from Training Set)
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="test")

# ✅ Compute Reward Model Accuracy
correct = 0
total = 0

def extract_text(conversation):
    return " ".join([turn["content"] for turn in conversation])

for example in dataset:
    chosen_text, rejected_text = example["chosen"], example["rejected"]
   
    chosen = extract_text(chosen_text)
    rejected = extract_text(rejected_text)
    
    # Tokenize both responses
    inputs_chosen = tokenizer(chosen, return_tensors="pt", padding=True, truncation=True)
    inputs_rejected = tokenizer(rejected, return_tensors="pt", padding=True, truncation=True)

    # Compute reward scores
    with torch.no_grad():
        score_chosen = reward_model(**inputs_chosen).logits.item()
        score_rejected = reward_model(**inputs_rejected).logits.item()

    # ✅ Reward Model should score the "chosen" response higher
    if score_chosen > score_rejected:
        correct += 1
    total += 1

# ✅ Compute Accuracy
accuracy = correct / total * 100
print(f"🏆 Reward Model Accuracy: {accuracy:.2f}%")
