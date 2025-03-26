from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ✅ Ensure the model loads from fine-tuned checkpoint
model_path = "./fine_tuned_gpt2"
print(f"Loading model from: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map="auto")

app = FastAPI()

class RequestData(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(data: RequestData):
    inputs = tokenizer(data.prompt, return_tensors="pt", truncation=True, max_length=256)
    outputs = model.generate(**inputs, max_length=200)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": response_text}

print("✅ API is ready!")
