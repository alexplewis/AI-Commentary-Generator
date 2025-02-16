import torch
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

# Load structured event dataset
df = pd.read_csv("data/event_rewriting_data.csv").dropna()
df = df[df["natural_description"] != ""]  # Remove empty labels

# Split into training & validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["structured_event"].tolist(),
    df["natural_description"].tolist(),
    test_size=0.1, random_state=42
)

# Load T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Tokenize the datasets
def tokenize_data(examples):
    return tokenizer(examples["structured_event"], text_target=examples["natural_description"], truncation=True, padding="max_length", max_length=128)

# Convert to Hugging Face dataset
train_dataset = Dataset.from_dict({"structured_event": train_texts, "natural_description": train_labels}).map(tokenize_data, batched=True)
val_dataset = Dataset.from_dict({"structured_event": val_texts, "natural_description": val_labels}).map(tokenize_data, batched=True)

# Load pre-trained T5 model
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./t5-event-rewriter",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    num_train_epochs=3,
    learning_rate=3e-4,
    weight_decay=0.01,
    save_total_limit=1,
    push_to_hub=False
)

# Initialize Trainer
trainer = Trainer(
    model=model.to(device),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

# Train the model
trainer.train()

# Save model and tokenizer
model.save_pretrained("./t5-event-rewriter")
tokenizer.save_pretrained("./t5-event-rewriter")

print("✅ Event rewriting model training complete. Saved in ./t5-event-rewriter")
