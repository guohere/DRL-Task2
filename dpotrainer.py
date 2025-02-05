from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import DPOConfig, DPOTrainer

# ✅ Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-SFT/checkpoint-395")

# ✅ Configure Efficient 4-bit Quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  
    bnb_4bit_compute_dtype="float16",  # ✅ Ensure efficient computation
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# ✅ Load Model with Memory Efficiency
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-SFT/checkpoint-395",
    quantization_config=quantization_config,
    device_map="auto"
)

# ✅ Disable `use_cache` to avoid conflicts
model.config.use_cache = False  

# ✅ Apply LoRA (Low-Rank Adaptation) for Fine-Tuning
lora_config = LoraConfig(
    r=16,                 # ✅ Reduced rank for lower memory
    lora_alpha=16,        # ✅ Scaling factor
    lora_dropout=0.1,     # ✅ Prevent overfitting
    target_modules=["q_proj", "v_proj"],  # ✅ Optimize attention layers
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ✅ Load and Reduce Dataset Size
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:15%]")  # ✅ Use 5% of dataset to reduce steps

# ✅ Configure Efficient DPO Training
training_args = DPOConfig(
    output_dir="Qwen2.5-0.5B-DPO",
    per_device_train_batch_size=1,   # ✅ Reduce batch size to fit memory
    gradient_accumulation_steps=4,   # ✅ Lower accumulation to reduce steps
    gradient_checkpointing=True,     # ✅ Save memory
    optim="paged_adamw_32bit",       # ✅ Optimized for LoRA & quantization
    bf16=True,                       # ✅ Use `bfloat16` if supported
    logging_steps=25,
    num_train_epochs=1,
    save_strategy="epoch",
    max_steps=500                    # ✅ Limit training steps to reduce total steps (Adjustable)
)

# ✅ Initialize DPO Trainer
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer
)

# ✅ Start Training
trainer.train()
