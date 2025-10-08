import warnings

import pandas as pd
import torch
from datasets import Dataset
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

print(torch.cuda.is_available())  # Should return: True
print(torch.cuda.get_device_name(0))  # Should print: "NVIDIA GeForce RTX 5090" (or similar)
print(torch.cuda.current_device())  # Should return: 0
print(torch.cuda.device_count())  # Should return: 1 (or more if multi-GPU)

# Load dataset
df = pd.read_csv("resources/csv_files/go_emotions_dataset.csv")

# Remove unnecessary columns
df = df.drop(columns=["id", "example_very_unclear"])

# Get emotion columns (multi-label)
emotion_columns = df.columns[1:]
df[emotion_columns] = df[emotion_columns].astype(int)  # Ensure binary labels
df["labels"] = df[emotion_columns].values.tolist()  # Convert to lists

# Keep only text and labels
df = df[["text", "labels"]]

# Split into train and test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(), df["labels"].tolist(), test_size=0.2, random_state=42
)

# Convert labels to tensors
train_labels = torch.tensor(train_labels, dtype=torch.float)
test_labels = torch.tensor(test_labels, dtype=torch.float)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_dict({"text": train_texts, "labels": train_labels})
test_dataset = Dataset.from_dict({"text": test_texts, "labels": test_labels})

# Load pre-trained tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)


# Apply tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove original text column
train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])

# Set dataset format for PyTorch
train_dataset.set_format("torch")
test_dataset.set_format("torch")


# Define custom model with correct loss function
class MultiLabelBert(BertForSequenceClassification):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = BCEWithLogitsLoss()
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# Initialize model
num_labels = len(emotion_columns)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiLabelBert.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels,
    problem_type="multi_label_classification"
).to(device)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./bert_emotions_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True
)


# Compute Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    predictions = (probs >= 0.5).astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
        precision = precision_score(labels, predictions, average='micro', zero_division=0)
        recall = recall_score(labels, predictions, average='micro', zero_division=0)

    return {"f1_micro": f1_micro, "precision": precision, "recall": recall}


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train model
trainer.train()

# Save model & tokenizer
trainer.save_model("./fine_tuned_bert_emotions")
tokenizer.save_pretrained("./fine_tuned_bert_emotions")


# Prediction Function
def predict_emotion(text, model, tokenizer, emotion_labels):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

    with torch.no_grad():
        outputs = model(**tokens)

    probs = torch.sigmoid(outputs.logits.detach()).cpu().numpy()
    predictions = (probs >= 0.5).astype(int)

    return [emotion_labels[i] for i, val in enumerate(predictions[0]) if val == 1]


# Test
emotion_labels = list(emotion_columns)
print(predict_emotion("I feel so sad and lonely today.", model, tokenizer, emotion_labels))
print(predict_emotion("I just got a new job! I'm so happy.", model, tokenizer, emotion_labels))



#############

from transformers import BertForSequenceClassification

# tokenizer = AutoTokenizer.from_pretrained("fine_tuned_bert_emotions")
tokenizer = AutoTokenizer.from_pretrained("sdeakin/fine_tuned_bert_emotions")

# bert_model = BertForSequenceClassification.from_pretrained("fine_tuned_bert_emotions", num_labels=num_labels)

# Load BERT model with sigmoid activation for multi-label classification
bert_model = BertForSequenceClassification.from_pretrained(
    "sdeakin/fine_tuned_bert_emotions",
    num_labels=num_labels,
    problem_type="multi_label_classification",
    ignore_mismatched_sizes=True
)

from transformers import BertTokenizer


# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def encode_texts(texts, tokenizer, max_length=512):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_length
    )


def get_bert_embeddings(text_list, batch_size=128):
    """Tokenize text and extract BERT embeddings ensuring all samples are processed."""
    embeddings = []

    total_batches = len(text_list) // batch_size + int(len(text_list) % batch_size != 0)

    print(f" Total texts to process: {len(text_list)}")
    print(f" Total batches expected: {total_batches}")

    with torch.no_grad():
        for i in tqdm(range(0, len(text_list), batch_size), desc="Processing BERT Embeddings"):
            if i + batch_size > len(text_list):  # Ensure last batch is fully processed
                batch_texts = text_list[i:]
            else:
                batch_texts = text_list[i: i + batch_size]

            # Tokenize batch
            # tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            tokens = tokenizer(batch_texts, padding="max_length", truncation=True, return_tensors="pt", max_length=64)

            # Move tokens to GPU if available
            # tokens = {key: value.to(device) for key, value in tokens.items()}

            # Extract embeddings from BERT (use .bert to avoid classification head)
            outputs = bert_model.bert(**tokens)

            # Mean pooling: Average all token embeddings
            # batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            batch_embeddings = outputs.last_hidden_state.cpu().numpy()  # shape: [batch, seq_len, 768]
            embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings)  # Stack all batches

    # **Check final shape**
    print(f" Expected embeddings: {len(text_list)}, Extracted embeddings: {embeddings.shape[0]}")

    return embeddings


# Convert training and test texts into BERT embeddings
print(f"\nConverting training and test sets into BERT embeddings...")
X_train_bert = get_bert_embeddings(train_texts, batch_size=32)
X_test_bert = get_bert_embeddings(test_texts, batch_size=32)
