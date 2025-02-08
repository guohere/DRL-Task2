### Google Colab Notebook: RLHF for Toxicity Reduction in AI Text Generation

# ✅ Install Required Libraries

# ✅ Import Libraries
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig
import torch

# ✅ Load Tokenizer
model_name = "Qwen/Qwen2.5-0.5B-SFT"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Load Pretrained Model (LLM)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
model.config.use_cache = False  # Disable cache for fine-tuning

# ✅ Define Sample Human Feedback Dataset
sample_data = [
    {"prompt": "Tell me about vaccines", "chosen": "Vaccines save lives and prevent diseases.", "rejected": "Vaccines are harmful."},
    {"prompt": "How to lose weight?", "chosen": "A balanced diet and regular exercise help with weight loss.", "rejected": "Just stop eating."}
]

dataset = load_dataset("json", data_files={"train": sample_data})

dataset = dataset["train"]
print("Sample dataset loaded:", dataset[:2])

# ✅ Train Reward Model (Classifies Chosen vs. Rejected Responses)
reward_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to("cuda")

# ✅ Apply LoRA Adaptation for Efficient Fine-Tuning
lora_config = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.1, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ✅ Configure DPO Training
training_args = DPOConfig(
    output_dir="rlhf_toxicity_model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    bf16=torch.cuda.is_bf16_supported(),  # Use bf16 if supported, otherwise fp16
    fp16=not torch.cuda.is_bf16_supported()
)

# ✅ Train Model with DPO
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)
trainer.train()

# ✅ Evaluate the Model (Before vs. After RLHF)
def generate_response(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_length=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

sample_prompt = "Tell me about vaccines"
print("\nBefore RLHF:")
print(generate_response(sample_prompt, model, tokenizer))

print("\nAfter RLHF:")
print(generate_response(sample_prompt, trainer.model, tokenizer))
