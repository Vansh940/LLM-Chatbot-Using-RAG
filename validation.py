import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

print("Loading model...")

base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "./chatbot_model"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(base_model)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    dtype=torch.float16
)

model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

print("Model loaded.")

# Load validation dataset
print("Loading validation dataset...")
df = pd.read_parquet("validation.parquet")

print("Validation samples:", len(df))

correct = 0
total = 0

for i in range(min(50, len(df))):   # test first 50 examples

    question = df.iloc[i]["text"]

    prompt = f"<user>: {question}\n<assistant>:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    answer = response.split("<assistant>:")[-1]

    print("\nQuestion:", question)
    print("Model:", answer)

    total += 1

print("\nEvaluation finished.")
print("Samples tested:", total)