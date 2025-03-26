import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

import os
os.environ["WANDB_DISABLED"] = "true"  # Disable Weights & Biases logging

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1️⃣ Load the CSV File
csv_file = "telecom_expenses.csv"  # Update with actual path if needed
df = pd.read_csv(csv_file)

# 2️⃣ Convert Data into Text Format for LLM
def row_to_text(row):
    return (
        f"In {row['Year']}, {row['Company']} had a total telecom expense of ${row['Total_Expense']}. "
        f"Call costs were ${row['Call_Cost']}, and data usage was {row['Data_Usage_GB']}GB."
    )

df["text_data"] = df.apply(row_to_text, axis=1)

# Display the transformed data
print(df["text_data"].head())

# 🔹 Save Processed Text Data
processed_text_file = "processed_expenses.txt"
df["text_data"].to_csv(processed_text_file, index=False, header=False)

print("✅ Data Converted to Text Format")


# 3️⃣ Load GPT-2 Model
model_id = "gpt2"  # Change to "gpt2-medium" or "gpt2-large" for better results

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a padding token

# Load the model with precision adjustment and device mapping
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32 if device == "cpu" else torch.float16,  # Use float16 for GPU
    device_map="auto"
)

print("✅ GPT-2 Model Loaded Successfully!")


# 4️⃣ Load processed data as a Dataset
dataset = Dataset.from_pandas(df[["text_data"]])

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text_data"], truncation=True, padding="max_length", max_length=256)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

print("✅ Data Tokenized Successfully!")

# 5️⃣ Define Training Arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_gpt2",
    per_device_train_batch_size=2,  # Adjust based on GPU memory
    num_train_epochs=3,
    save_steps=500,
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True if device == "cuda" else False,  # Mixed precision for GPUs
    push_to_hub=False  # Set to True if you want to upload to Hugging Face Hub
)

# Data collator (handles padding & formatting)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 6️⃣ Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 7️⃣ Start Training
trainer.train()

print("✅ Fine-Tuning Complete!")

# 8️⃣ Save Model
model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")

print("✅ Model saved in './fine_tuned_gpt2'")
