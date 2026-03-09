import torch

print("=====================================")
print("Checking GPU...")
print("GPU AVAILABLE:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")

print("=====================================")

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model

print("======================================")
print("LLM Chatbot Training Script Started")
print("======================================\n")

# ---------------------------
# Step 1: Load Model
# ---------------------------
print("[1/6] Loading base model...")

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# TinyLlama doesn't have pad token
tokenizer.pad_token = tokenizer.eos_token

print("Setting 4-bit quantization...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

print("Model loaded successfully!\n")

# Enable gradient checkpointing (saves VRAM)
model.gradient_checkpointing_enable()

# ---------------------------
# Step 2: Load Dataset
# ---------------------------
print("[2/6] Loading processed dataset...")

dataset = load_dataset("json", data_files="chat_data.json")

print("Dataset loaded.")
print("Training samples:", len(dataset["train"]), "\n")

# ---------------------------
# Step 3: Tokenization
# ---------------------------
print("[3/6] Tokenizing dataset...")

def format_chat(example):

    text = (
        "<user>: " + example["messages"][0]["content"] +
        "\n<assistant>: " + example["messages"][1]["content"]
    )

    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512
    )

    tokens["labels"] = tokens["input_ids"].copy()

    return tokens

dataset = dataset.map(format_chat, remove_columns=dataset["train"].column_names)

print("Tokenization completed!\n")

# ---------------------------
# Step 4: Setup LoRA
# ---------------------------
print("[4/6] Configuring LoRA fine-tuning...")

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

print("LoRA configuration applied!\n")

# ---------------------------
# Step 5: Data Collator
# ---------------------------
print("[5/6] Creating data collator...")

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

# ---------------------------
# Step 6: Training Setup
# ---------------------------
print("[6/6] Preparing training configuration...")

training_args = TrainingArguments(
    output_dir="./chatbot_model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=500,
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator
)

print("Trainer initialized.")
print("Starting training...\n")

trainer.train()

print("\nTraining finished successfully!")

print("Saving trained model...")

trainer.save_model("./chatbot_model")
tokenizer.save_pretrained("./chatbot_model")

print("Model saved in ./chatbot_model")
print("Training pipeline complete.")
print("======================================")