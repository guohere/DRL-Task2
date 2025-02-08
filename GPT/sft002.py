import os
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForLanguageModeling


# 1️⃣ Load and Sample Dataset
dataset_name = "CarperAI/openai_summarize_tldr"
raw_datasets = load_dataset(dataset_name)

# Select 5% of training data and 10% of validation data
train_percentage = 0.05  # 5%
valid_percentage = 0.10  # 10%

# Shuffle and randomly select subsets
train_dataset = raw_datasets["train"].shuffle(seed=42).select(range(int(len(raw_datasets["train"]) * train_percentage)))
eval_dataset = raw_datasets["valid"].shuffle(seed=42).select(range(int(len(raw_datasets["valid"]) * valid_percentage)))

# 2️⃣ Load Tokenizer & Model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 requires this for padding

config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, config=config)

# 3️⃣ Preprocessing Function (✅ Fix: Ensures Padding & Truncation)
def preprocess_function(examples):
    inputs = [f"Summarize the following:\n{p}\nSummary:" for p in examples["prompt"]]
    targets = examples["label"]

    # ✅ Fix: Ensure labels are tokenized correctly
    model_inputs = tokenizer(
        inputs, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
    )
    
    labels = tokenizer(
        targets, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
    )["input_ids"]

    # ✅ Fix: Convert padding token (usually 50256 for GPT-2) to -100 for label ignoring in loss calculation
    labels[labels == tokenizer.pad_token_id] = -100

    model_inputs["labels"] = labels
    return model_inputs

# Apply preprocessing
train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)

# ✅ Fix: Use proper data collator for GPT-2
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # MLM (Masked Language Modeling) is False since we are using causal LM
)

# 4️⃣ Define Training Configuration
training_args = SFTConfig(
    output_dir="GPT-2-SFT",  # Save location
    per_device_train_batch_size=4,  # Increase batch size if GPU memory allows
    gradient_accumulation_steps=8,  # For larger effective batch size
    gradient_checkpointing=True,  # Saves memory
    num_train_epochs=3,  # Number of training epochs
    save_strategy="steps",
    save_steps=500,  # Save every 500 steps
    evaluation_strategy="steps",
    eval_steps=250,  # Evaluate every 250 steps
    logging_steps=100,  # Log training progress every 100 steps
    bf16=True,  # Enable mixed-precision training (use False if no BF16 support)
    optim="paged_adamw_32bit",  # Optimizer suited for memory efficiency
    warmup_steps=500,  # Learning rate warmup steps
    learning_rate=5e-5,  # Standard fine-tuning LR for GPT-2
    weight_decay=0.01,  # Helps with generalization
)

# 5️⃣ Initialize Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,  # ✅ Fix: Use correct collator
    processing_class=tokenizer,  # ✅ Fix: Use processing_class instead of tokenizer
)

# 6️⃣ Train Model
trainer.train()

# 7️⃣ Track & Plot Training Loss
train_loss = []
eval_loss = []
steps = []

for log in trainer.state.log_history:
    if "loss" in log:  # ✅ Track Training Loss
        train_loss.append(log["loss"])
        steps.append(log["step"])
    if "eval_loss" in log:  # ✅ Track Validation Loss
        eval_loss.append(log["eval_loss"])

# Plot Training vs Validation Loss
print("📊 Saving Training Progress Plot...")

plt.figure(figsize=(8, 6))
plt.plot(steps, train_loss, label="Training Loss", color="red", linestyle="dashed", marker="o")
plt.plot(steps[:len(eval_loss)], eval_loss, label="Validation Loss", color="blue", marker="s")

plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training Loss & Validation Loss Over Time")
plt.legend()
plt.grid()

# Save & Show Plot
plt.savefig("training_progress.png")
plt.show()

# 8️⃣ Save Final Model
output_dir = "GPT-2-SFT"
os.makedirs(output_dir, exist_ok=True)
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ Supervised Fine-Tuning complete! Model saved to: {output_dir}")
print("✅ Training loss plot saved as `training_progress.png`.")
