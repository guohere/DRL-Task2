

"""
DPO evaluation:
base model, SFT model and DPO model comparsion
"""

import torch
import math
from datasets import load_metric
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

# ✅ Load Tokenizers
tokenizer_base = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer_sft = AutoTokenizer.from_pretrained("Qwen2.5-0.5B-SFT/checkpoint-1975")
tokenizer_dpo = AutoTokenizer.from_pretrained("Qwen2.5-0.5B-DPO/checkpoint-5000")
reward_tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-0.5B-Reward/checkpoint-182")

# ✅ Load Models
model_base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
model_sft = AutoModelForCausalLM.from_pretrained("Qwen2.5-0.5B-SFT/checkpoint-1975")
model_dpo = AutoModelForCausalLM.from_pretrained("Qwen2.5-0.5B-DPO/checkpoint-5000")
reward_model = AutoModelForSequenceClassification.from_pretrained("Qwen2.5-0.5B-Reward/checkpoint-182")

# ✅ Define a test prompt
test_prompt = "How can AI improve climate change policies?"

# -------------------------
# 1️⃣ Compute Perplexity
# -------------------------
def compute_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss.item()
    return math.exp(loss)

print("\n📊 Evaluating Perplexity...")
ppl_base = compute_perplexity(model_base, tokenizer_base, test_prompt)
ppl_sft = compute_perplexity(model_sft, tokenizer_sft, test_prompt)
ppl_dpo = compute_perplexity(model_dpo, tokenizer_dpo, test_prompt)

print(f"📊 Perplexity (Base Model): {ppl_base:.2f}")
print(f"📊 Perplexity (SFT Model): {ppl_sft:.2f}")
print(f"📊 Perplexity (DPO Model): {ppl_dpo:.2f}")

# -------------------------
# 2️⃣ Generate & Compare Responses
# -------------------------
def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

print("\n📝 Generating Responses...")
response_base = generate_response(model_base, tokenizer_base, test_prompt)
response_sft = generate_response(model_sft, tokenizer_sft, test_prompt)
response_dpo = generate_response(model_dpo, tokenizer_dpo, test_prompt)

print(f"\n🔹 Base Model Response: {response_base}\n")
print(f"🔹 SFT Model Response: {response_sft}\n")
print(f"🔹 DPO Model Response: {response_dpo}\n")

# -------------------------
# 3️⃣ Use Reward Model to Score Responses
# -------------------------
def get_reward_score(model, tokenizer, response):
    inputs = tokenizer(response, return_tensors="pt")
    with torch.no_grad():
        score = model(**inputs).logits.item()
    return score

print("\n🏆 Scoring Responses with Reward Model...")
reward_sft = get_reward_score(reward_model, reward_tokenizer, response_sft)
reward_dpo = get_reward_score(reward_model, reward_tokenizer, response_dpo)

print(f"🏆 Reward Score (SFT Model): {reward_sft:.2f}")
print(f"🏆 Reward Score (DPO Model): {reward_dpo:.2f}")


# -------------------------
# 🎯 Final Summary
# -------------------------
print("\n📊 Final Evaluation Summary:")
print(f"📊 Perplexity (Base/SFT/DPO): {ppl_base:.2f} / {ppl_sft:.2f} / {ppl_dpo:.2f}")
print(f"🏆 Reward Score (SFT/DPO): {reward_sft:.2f} / {reward_dpo:.2f}")

print("\n✅ Evaluation Complete!")
