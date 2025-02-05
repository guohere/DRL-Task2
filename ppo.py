import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig

# ✅ Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-SFT/checkpoint-395")

# ✅ Load Policy Model (from SFT)
policy_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-SFT/checkpoint-395",
    load_in_4bit=True,
    device_map="auto"
)

# ✅ Load Reference Model (Same as SFT, frozen)
reference_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-SFT/checkpoint-395",
    load_in_4bit=True,
    device_map="auto"
)

# ✅ Load Reward Model
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen2.5-0.5B-Reward/checkpoint-182",
    num_labels=1
)

value_model = policy_model

# ✅ Load Dataset (for RL training)
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train["train_gen"][:1%]")

print("🔍 Dataset Structure BEFORE Mapping:")
print(dataset.column_names)

def preprocess_function(examples):
    # ✅ Extract text content safely
    chosen_texts = [" ".join([turn["content"] for turn in example if "content" in turn]) for example in examples["chosen"]]
    rejected_texts = [" ".join([turn["content"] for turn in example if "content" in turn]) for example in examples["rejected"]]

    # ✅ Tokenize correctly in batch mode
    chosen_tokens = tokenizer(chosen_texts, truncation=True, padding="max_length", return_tensors="pt")
    rejected_tokens = tokenizer(rejected_texts, truncation=True, padding="max_length", return_tensors="pt")

    return {
        "input_ids": chosen_tokens["input_ids"].tolist(),  # ✅ Convert to list of lists
        "attention_mask": chosen_tokens["attention_mask"].tolist(),
        "input_ids_rejected": rejected_tokens["input_ids"].tolist(),
        "attention_mask_rejected": rejected_tokens["attention_mask"].tolist(),
    }

# ✅ Apply dataset mapping with `batched=True`
dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)


print("\n🔢 Number of Elements Per Column After Mapping:")
for column in dataset.column_names:
    print(f"{column}: {len(dataset[column])}")

# ✅ Configure PPO Training
training_args = PPOConfig(
    output_dir="Qwen2.5-0.5B-PPO",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    bf16=True,
    logging_steps=25,
    num_train_epochs=1,
    save_strategy="epoch"
)

# ✅ Initialize PPO Trainer
trainer = PPOTrainer(
    model=policy_model,
    ref_model=reference_model,  # Reference model remains unchanged
    args=training_args,
    train_dataset=dataset,
    value_model=value_model,
    reward_model=reward_model,
    tokenizer=tokenizer
)

# ✅ Train PPO Model
trainer.train()

# ✅ Save Final PPO Model
trainer.save_model("Qwen2.5-0.5B-PPO/final_checkpoint")
