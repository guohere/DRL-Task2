from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math

def compute_perplexity(model_name, test_text):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

    inputs = tokenizer(test_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

    perplexity = math.exp(loss.item())
    return perplexity

test_text = "The theory of relativity is one of the most important scientific discoveries of the 20th century."
ppl_sft = compute_perplexity("Qwen/Qwen2.5-0.5B-SFT/checkpoint-395", test_text)
ppl_base = compute_perplexity("Qwen/Qwen2.5-0.5B", test_text)

print(f"📊 Perplexity (Base Model): {ppl_base:.2f}")
print(f"📊 Perplexity (Fine-Tuned Model): {ppl_sft:.2f}")
