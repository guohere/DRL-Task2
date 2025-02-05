from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ✅ Load Pre-trained Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

# ✅ Define a Sample Prompt
prompt = "How can AI help in climate change?"
inputs = tokenizer(prompt, return_tensors="pt")

# ✅ Generate Two Different Responses (Simulating Fine-Tuning Variants)
with torch.no_grad():
    outputs1 = model.generate(**inputs, max_length=100)
    outputs2 = model.generate(**inputs, max_length=100)

response1 = tokenizer.decode(outputs1[0], skip_special_tokens=True)
response2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)

print(f"🔹 Response 1: {response1}\n")
print(f"🔹 Response 2: {response2}\n")

# ✅ Simulated Reward Scores (Higher = Preferred)
reward1 = 3.5  # Arbitrary score for Response 1
reward2 = 4.7  # Response 2 is preferred

preferred_response = response1 if reward1 > reward2 else response2
print(f"✅ Preferred Response (Higher Reward): {preferred_response}\n")

# ✅ Simulate Reinforcement Learning Update
reinforced_prompt = f"{prompt}\n[Better AI-generated response based on feedback]: {preferred_response}"

# ✅ Generate Final Optimized Response
inputs = tokenizer(reinforced_prompt, return_tensors="pt")
with torch.no_grad():
    final_output = model.generate(**inputs, max_length=100)

final_response = tokenizer.decode(final_output[0], skip_special_tokens=True)
print(f"🚀 Final Optimized Response: {final_response}")
