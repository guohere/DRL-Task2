from trl import RewardConfig, RewardTrainer
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch

# ✅ Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-SFT/checkpoint-1975")

# ✅ Load QLoRA 4-bit Quantization Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # ✅ Better stability than fp16
    bnb_4bit_use_double_quant=True,
)

# ✅ Load Model with Quantization
model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen2.5-0.5B-SFT/checkpoint-1975",
    num_labels=1,
    quantization_config=bnb_config,  
    device_map="auto",
)

# ✅ Ensure Pad Token is Correct
model.config.pad_token_id = tokenizer.pad_token_id

# ✅ Apply LoRA for Efficient Fine-Tuning
lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS",  # ✅ Required for reward models
)

# ✅ Convert to LoRA model
model = get_peft_model(model, lora_config)

# ✅ Ensure Training Mode
model.train()

# ✅ Disable `use_cache` for Gradient Checkpointing
model.config.use_cache = False

# ✅ Load Dataset (Use a Subset to Reduce Training Time)
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:50%]")  # ✅ Use only 20% for speed

# ✅ Optimized Training Parameters
training_args = RewardConfig(
    output_dir="Qwen2.5-0.5B-Reward",
    per_device_train_batch_size=4,  # ✅ Small batch size for memory efficiency
    gradient_accumulation_steps=8,  # ✅ Reduces memory use
    num_train_epochs=1,  # ✅ Less training time
    gradient_checkpointing=True,  # ✅ Saves memory
    bf16=True,  # ✅ More stable than fp16
    optim="paged_adamw_32bit",
    save_strategy="epoch",
    report_to="none"
)

#    attn_implementation="flash_attention_2",  # ✅ Optional, speeds up training (~20%)

# ✅ Initialize Trainer
trainer = RewardTrainer(
    args=training_args,
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
)

# ✅ Debug: Check Tensor Types
inputs = tokenizer("Test input", return_tensors="pt")
inputs["labels"] = torch.tensor([1.0]).to("cuda")  # ✅ Reward model needs float labels
inputs = {key: value.to("cuda") for key, value in inputs.items()}

# ✅ Forward Pass Check
outputs = model(**inputs)
loss = outputs.get("loss", None)

if loss is None:
    raise RuntimeError("🚨 Loss is None! Ensure labels are correctly formatted.")
elif not loss.requires_grad:
    raise RuntimeError("🚨 Loss does not have gradients! Check LoRA setup.")

# ✅ Start Training
trainer.train()
