import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

MODEL_NAME = "ProsusAI/finbert"
INPUT_CSV = "data/finbert_training_data.csv"
OUTPUT_DIR = "./model/finbert_finetuned_local"
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 3

df = pd.read_csv(INPUT_CSV)
df = df.dropna(subset=['text', 'sentiment'])

# Encode Labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['sentiment'])

# Train/Val Split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to HF Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# --- 2. Tokenization ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_LEN
    )

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)


cols_to_remove = ['text', 'Date', 'sentiment', 'confidence', '__index_level_0__']

# Safely remove columns that exist
tokenized_train = tokenized_train.remove_columns([c for c in cols_to_remove if c in tokenized_train.column_names])
tokenized_val = tokenized_val.remove_columns([c for c in cols_to_remove if c in tokenized_val.column_names])

tokenized_train.set_format("torch")
tokenized_val.set_format("torch")

# --- 3. Model Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=len(label_encoder.classes_)
).to(device)

# --- 4. Training ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    save_safetensors=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)

trainer.train()

# --- 5. Save Model ---
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Model saved to {OUTPUT_DIR}")