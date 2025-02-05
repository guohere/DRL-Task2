from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch

# ✅ Load dataset
dataset = load_dataset("trl-lib/Capybara", split="train[:20%]")

# ✅ Configure 4-bit quantization for memory-efficient training
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # 🔥 Use `bfloat16` (better than `fp16`)
    bnb_4bit_use_double_quant=True,
)

# ✅ Load quantized model
model_name = "Qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,  # Apply QLoRA optimization
    device_map="auto",
)

# ✅ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Apply LoRA adapters for fine-tuning
lora_config = LoraConfig(
    r=16,  # LoRA rank (reduces memory use)
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Dropout for stability
    bias="none",
    task_type="CAUSAL_LM",
)

# Convert to LoRA model
model = get_peft_model(model, lora_config)

# 🔥 Fix: Ensure model is in training mode
model.train()

# 🔥 Fix: Disable `use_cache` (incompatible with gradient checkpointing)
model.config.use_cache = False

# 🔥 Fix: Ensure only LoRA layers require gradients
for name, param in model.named_parameters():
    if "lora" in name and param.dtype in [torch.float16, torch.float32, torch.bfloat16]:  
        param.requires_grad = True
        print(f"✅ Trainable: {name}")
    else:
        param.requires_grad = False  # ✅ Freeze base model

# 🔥 Debug: Ensure all trainable parameters actually require gradients
trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
if not trainable_params:
    raise RuntimeError("🚨 No trainable parameters detected! Check LoRA setup.")

# ✅ Configure training parameters
training_args = SFTConfig(
    output_dir="Qwen/Qwen2.5-0.5B-SFT",
    per_device_train_batch_size=1,  # Reduce batch size to fit in memory
    gradient_accumulation_steps=8,  # Accumulate gradients to reduce memory usage
    gradient_checkpointing=False,  # 🚨 Temporarily disabled to debug
    num_train_epochs=1,
    logging_steps=25,
    bf16=True,  # ✅ Use `bfloat16` (Fix FP16 issues)
    optim="paged_adamw_32bit",  # ✅ Optimizer optimized for QLoRA
    report_to="none",
)

# ✅ Initialize trainer
trainer = SFTTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset,
)

# ✅ Debugging: Check if model outputs require gradients
inputs = tokenizer("Test input", return_tensors="pt")
inputs["labels"] = inputs["input_ids"].detach().clone()  # ✅ Ensure labels are included
inputs = {key: value.to("cuda") for key, value in inputs.items()}  # Move to GPU

# 🔥 Fix: Convert `input_ids` and `labels` to `torch.long`
inputs["input_ids"] = inputs["input_ids"].long()
inputs["labels"] = inputs["labels"].long()

outputs = model(**inputs)  # Forward pass

# ✅ Check if logits require gradients
if hasattr(outputs, "logits"):
    print(f"🚀 Model Output Requires Grad: {outputs.logits.requires_grad}")

# ✅ Extract the loss correctly
loss = outputs.get("loss", None)
if loss is None:
    raise RuntimeError("🚨 Loss is None! Model is not computing loss correctly.")
elif not loss.requires_grad:
    raise RuntimeError("🚨 Loss does not have gradients! Something is wrong with input setup.")

# ✅ Start training
trainer.train()
