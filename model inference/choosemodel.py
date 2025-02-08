import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ✅ Load Models
models = {
    "base": (
        AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").to("cuda"),
        AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B"),
    ),
    "sft": (
        AutoModelForCausalLM.from_pretrained("Qwen2.5-0.5B-SFT/checkpoint-1975").to("cuda"),
        AutoTokenizer.from_pretrained("Qwen2.5-0.5B-SFT/checkpoint-1975"),
    ),
    "dpo": (
        AutoModelForCausalLM.from_pretrained("Qwen2.5-0.5B-DPO/checkpoint-15533").to("cuda"),
        AutoTokenizer.from_pretrained("Qwen2.5-0.5B-DPO/checkpoint-15533"),
    ),
}

def chat_with_model(prompt, model_name):
    if model_name not in models:
        return "Invalid model choice. Choose from: base, sft, dpo."

    model, tokenizer = models[model_name]
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=512, do_sample=True, temperature=0.7, top_p=0.9)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

gr.Interface(
    fn=chat_with_model,
    inputs=["text", gr.Radio(["base", "sft", "dpo"], label="Choose Model")],
    outputs="text",
    title="🤖 AI Chatbot with Multiple Models",
    description="Select a model and chat with it!",
).launch(share=True)
