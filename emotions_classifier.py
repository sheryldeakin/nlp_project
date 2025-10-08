import pandas as pd

# Load dataset
df = pd.read_csv("go_emotions_dataset.csv")  # Update with the correct path

# Display first few rows
print(df.head())

# Remove unnecessary columns
df = df.drop(columns=["id", "example_very_unclear"])

# Get emotion columns
emotion_columns = df.columns[1:]

# Assign single label based on the highest scoring emotion
df["label"] = df[emotion_columns].idxmax(axis=1)

# Keep only text and label
df = df[["text", "label"]]

# Show updated dataset
df.head()

# For initial testing, smaller sample
df_sample = df.sample(n=50000, random_state=42)  # Take 10,000 random samples

from sklearn.model_selection import train_test_split

#  # Use smaller dataset for training
# train_texts, test_texts, train_labels, test_labels = train_test_split(
#     df_sample["text"], df_sample["label"], test_size=0.2, random_state=42
# )

# # Split dataset (80% train, 20% test)
# train_texts, test_texts, train_labels, test_labels = train_test_split(
#     df["text"], df["label"], test_size=0.2, random_state=42
# )

###########################################################################################
# Pre-Cleaning
###########################################################################################
import contractions
import re
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

slang_dict = {"brb": "be right back", "idk": "I don't know", "u": "you"}


def clean_text(text):
    re_number = re.compile('[0-9]+')
    re_url = re.compile("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    re_tag = re.compile('\[[A-Z]+\]')
    re_char = re.compile('[^0-9a-zA-Z\s?!.,:\'\"//]+')
    re_char_clean = re.compile('[^0-9a-zA-Z\s?!.,\[\]]')
    re_punc = re.compile('[?!,.\'\"]')

    text = re.sub(re_char, "", text)  # Remove unknown character
    text = contractions.fix(text)  # Expand contraction
    text = re.sub(re_url, ' [url] ', text)  # Replace URL with number
    text = re.sub(re_char_clean, "", text)  # Only alphanumeric and punctuations.
    # text = re.sub(re_punc, "", text) # Remove punctuation.
    text = text.lower()  # Lower text
    text = " ".join([w for w in text.split(' ') if w != " "])  # Remove whitespace

    #   text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    text = " ".join([slang_dict[word] if word in slang_dict else word for word in text.split()])

    # Makes BERT worse, but without BERT better
    #   text = " ".join([word for word in text.split() if word not in stop_words]) # Remove stopwords

    return text


# Apply text cleaning to dataset
df_sample["cleaned_text"] = df_sample["text"].apply(clean_text)

# Display first few rows to verify
print(df_sample[["text", "cleaned_text"]].head())

# Use smaller dataset for training
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df_sample["cleaned_text"], df_sample["label"], test_size=0.2, random_state=42
)

###########################################################################################
# TFID 
###########################################################################################

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")

# Transform text into TF-IDF vectors
X_train_tfidf = vectorizer.fit_transform(train_texts)
X_test_tfidf = vectorizer.transform(test_texts)

###########################################################################################
# Logistic Regression
###########################################################################################
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train Logistic Regression model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_tfidf, train_labels)

# Predict on test set
y_pred = clf.predict(X_test_tfidf)

# Evaluate accuracy
accuracy = accuracy_score(test_labels, y_pred)
print(f"Logistic Regression Accuracy: {accuracy:.2f}")

###########################################################################################
# SVM
###########################################################################################

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Try different kernels
kernels = ["linear", "poly", "rbf", "sigmoid"]

for kernel in kernels:
    print(f"\nTraining SVM with {kernel} kernel...")
    svm_clf = SVC(kernel=kernel, degree=3, gamma="scale")  # Set degree for poly
    svm_clf.fit(X_train_tfidf, train_labels)

    # Predict on test set
    y_pred_svm = svm_clf.predict(X_test_tfidf)

    # Evaluate accuracy
    accuracy_svm = accuracy_score(test_labels, y_pred_svm)
    print(f"SVM ({kernel}) Accuracy: {accuracy_svm:.2f}")

###########################################################################################
# Using BERT to expand text instead of TF-IDF
###########################################################################################
import numpy as np

from transformers import AutoTokenizer

# Load pre-trained BERT tokenizer & model - Baseline BERT model
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# bert_model = BertModel.from_pråetrained("bert-base-uncased")

from huggingface_hub import snapshot_download

model_path = snapshot_download(repo_id="sdeakin/bert_fine_tuned_emotions")

num_labels = df["label"].nunique()
from transformers import BertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(model_path)
bert_model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

# # Convert text into BERT embeddings using mean pooling
# def get_bert_embeddings(text_list, batch_size=32):
#     embeddings = []
#     for i in tqdm(range(0, len(text_list), batch_size), desc="Processing BERT Embeddings"):
#         batch_texts = text_list[i : i + batch_size]
#         tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

#         with torch.no_grad():
#             outputs = bert_model(**tokens)

#         # Mean pooling instead of using just [CLS] token
#         batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
#         embeddings.append(batch_embeddings)

#     return np.vstack(embeddings)

# For base model: Function to tokenize text and extract embeddings with a more detailed progress bar
# def get_bert_embeddings(text_list, batch_size=32):
#     """Tokenize text and extract BERT embeddings with a detailed progress bar."""
#     embeddings = []
#     total_batches = len(text_list) // batch_size + int(len(text_list) % batch_size != 0)

#     with tqdm(total=len(text_list), desc="Processing BERT Embeddings", unit="sentence") as pbar:
#         for i in range(0, len(text_list), batch_size):
#             batch_texts = text_list[i : i + batch_size]

#             # Tokenize batch
#             tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

#             with torch.no_grad():
#                 outputs = bert_model(**tokens)

#             # # Mean pooling: Average over all token embeddings
#             # batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
#             # Max Pooling (Alternative to Mean Pooling)
#             batch_embeddings = torch.max(outputs.last_hidden_state, dim=1)[0].cpu().numpy()
#             embeddings.append(batch_embeddings)

#             # Update progress bar by number of processed sentences
#             pbar.update(len(batch_texts))  

#     return np.vstack(embeddings)  # Combine all batches


# For custom bert model

print(f"Train texts: {len(train_texts)}")
print(f"Train labels: {len(train_labels)}")


def get_bert_embeddings(text_list, batch_size=128):
    """Tokenize text and extract BERT embeddings ensuring all samples are processed."""
    embeddings = []

    total_batches = len(text_list) // batch_size + int(len(text_list) % batch_size != 0)

    print(f"🔹 Total texts to process: {len(text_list)}")  # ✅ Debugging
    print(f"🔹 Total batches expected: {total_batches}")

    with torch.no_grad():
        for i in tqdm(range(0, len(text_list), batch_size), desc="Processing BERT Embeddings"):
            if i + batch_size > len(text_list):  # Ensure last batch is fully processed
                batch_texts = text_list[i:]
            else:
                batch_texts = text_list[i: i + batch_size]

            # Tokenize batch
            tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

            # Move tokens to GPU if available
            # tokens = {key: value.to(device) for key, value in tokens.items()}

            # Extract embeddings from BERT (use .bert to avoid classification head)
            outputs = bert_model.bert(**tokens)

            # Mean pooling: Average all token embeddings
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings)  # Stack all batches

    # **Check final shape**
    print(f" Expected embeddings: {len(text_list)}, Extracted embeddings: {embeddings.shape[0]}")

    return embeddings


# Convert training and test texts into BERT embeddings
print(f"\nConverting training and test sets into BERT embeddings...")
X_train_bert = get_bert_embeddings(train_texts.tolist(), batch_size=32)
X_test_bert = get_bert_embeddings(test_texts.tolist(), batch_size=32)

print("BERT embeddings shape:", X_train_bert.shape)  # Should be (num_samples, 768)

# # Verify correct alignment
# assert X_train_bert.shape[0] == len(train_labels), "Mismatch after BERT embedding!"
# assert X_test_bert.shape[0] == len(test_labels), "Mismatch after BERT embedding!"

# ###########################################################################################
# # BERT + Logistic Regression
# ###########################################################################################

# print("Starting Logistic Regression on BERT embeddings")
# # Train Logistic Regression model using BERT embeddings
# clf_bert = LogisticRegression(max_iter=1000)
# clf_bert.fit(X_train_bert, train_labels)

# # Predict on test set
# y_pred_bert = clf_bert.predict(X_test_bert)

# # Evaluate accuracy
# accuracy_bert = accuracy_score(test_labels, y_pred_bert)
# print(f"Logistic Regression (BERT) Accuracy: {accuracy_bert:.2f}")

###########################################################################################
# BERT + SGD
###########################################################################################
from tqdm import tqdm
from sklearn.linear_model import SGDClassifier  # Supports partial fit (online learning)

# Initialize SGD-based Logistic Regression (works iteratively)
clf_bert = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3)

# Convert labels to numerical (if needed)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Train in batches with a progress bar
num_epochs = 5  # Number of epochs
batch_size = 256  # Define batch size

print("\nTraining Logistic Regression with BERT Embeddings...\n")

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}:")
    for i in tqdm(range(0, X_train_bert.shape[0], batch_size), desc=f"Training Progress (Epoch {epoch + 1})"):
        batch_X = X_train_bert[i: i + batch_size]
        batch_y = train_labels_encoded[i: i + batch_size]

        # Perform batch-wise fitting
        clf_bert.partial_fit(batch_X, batch_y, classes=np.unique(train_labels_encoded))

# Predict on test set
y_pred_bert = clf_bert.predict(X_test_bert)

# Evaluate accuracy
accuracy_bert = accuracy_score(test_labels_encoded, y_pred_bert)
print(f"\nLogistic Regression (BERT) Accuracy: {accuracy_bert:.2f}")

###########################################################################################
# BERT + SVM Approximations to view progress
###########################################################################################
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler, PolynomialCountSketch
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Ensure tqdm works inside pandas apply() calls
tqdm.pandas()


# Function to display progress during feature transformation
def transform_with_progress(transformer, X, description="Transforming"):
    """Apply a transformation with a progress bar."""
    with tqdm(total=X.shape[0], desc=description, unit="samples") as pbar:
        X_transformed = transformer.fit_transform(X)
        pbar.update(X.shape[0])  # Mark all samples as processed
    return X_transformed


# Define different loss functions for SGDClassifier
loss_functions = {
    "Linear SVM": SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3),
    "Polynomial SVM": Pipeline([
        ("poly_features", PolynomialCountSketch(degree=3, random_state=42)),
        ("sgd", SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3))
    ]),
    "RBF SVM": Pipeline([
        ("rbf_features", RBFSampler(gamma=0.1, random_state=42)),
        ("sgd", SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3))
    ]),
    "Sigmoid SVM": SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3)
}

# # Simulated BERT embeddings (Use actual BERT embeddings in practice)
# X_train_bert = np.random.rand(8000, 768)  # 8000 samples, 768 features
# X_test_bert = np.random.rand(2000, 768)   # 2000 samples, 768 features
# train_labels_encoded = np.random.randint(0, 28, 8000)  # Random labels for simulation
# test_labels_encoded = np.random.randint(0, 28, 2000)

# Train and evaluate each model with progress bars
for name, model in loss_functions.items():
    print(f"\nTraining {name}...")

    # Apply feature transformation with progress bar if needed
    if isinstance(model, Pipeline):
        X_train_transformed = transform_with_progress(model.named_steps[list(model.named_steps.keys())[0]],
                                                      X_train_bert, description=f"Transforming Train Data ({name})")
        X_test_transformed = transform_with_progress(model.named_steps[list(model.named_steps.keys())[0]], X_test_bert,
                                                     description=f"Transforming Test Data ({name})")
        model.named_steps["sgd"].fit(X_train_transformed, train_labels_encoded)
    else:
        model.fit(X_train_bert, train_labels_encoded)

    # Training progress bar
    with tqdm(total=100, desc=f"Training {name}", unit="%") as pbar:
        for _ in range(100):
            pbar.update(1)  # Simulated update for training

    # Predict on test set
    if isinstance(model, Pipeline):
        y_pred_sgd = model.named_steps["sgd"].predict(X_test_transformed)
    else:
        y_pred_sgd = model.predict(X_test_bert)

    # Evaluate accuracy
    accuracy_sgd = accuracy_score(test_labels_encoded, y_pred_sgd)
    print(f"{name} Accuracy: {accuracy_sgd:.2f}")

# ###########################################################################################
# # Hybrid of TF-IDF + BERT on Logistic Regression
# ###########################################################################################
# from scipy.sparse import hstack

# # Combine TF-IDF features and BERT embeddings
# X_train_combined = hstack([X_train_tfidf, X_train_bert])
# X_test_combined = hstack([X_test_tfidf, X_test_bert])

# # Train Logistic Regression on combined features
# clf_combined = LogisticRegression(max_iter=1000)
# clf_combined.fit(X_train_combined, train_labels)

# # Predict on test set
# y_pred_combined = clf_combined.predict(X_test_combined)

# # Evaluate accuracy
# accuracy_combined = accuracy_score(test_labels, y_pred_combined)
# print(f"Combined (TF-IDF + BERT) Accuracy: {accuracy_combined:.2f}")

###########################################################################################
# SVM + BERT
###########################################################################################

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print(f"BERT Embeddings Shape: {X_train_bert.shape}")
print(f"Label Shape: {train_labels.shape}")

# # Ensure sizes match
# assert X_train_bert.shape[0] == len(train_labels), "Mismatch in training samples!"
# assert X_test_bert.shape[0] == len(test_labels), "Mismatch in test samples!"

# Try different kernels
kernels = ["linear", "poly", "rbf", "sigmoid"]

for kernel in kernels:
    print(f"\nTraining BERT + SVM with {kernel} kernel...")
    svm_clf = SVC(kernel=kernel, degree=3, gamma="scale")  # Set degree for poly
    svm_clf.fit(X_train_bert, train_labels)

    # Predict on test set
    y_pred_svm = svm_clf.predict(X_test_bert)

    # Evaluate accuracy
    accuracy_svm = accuracy_score(test_labels, y_pred_svm)
    print(f"SVM + BERT ({kernel}) Accuracy: {accuracy_svm:.2f}")

###########################################################################################
# MLP + BERT
###########################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Convert labels to numerical format
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Convert resources to PyTorch tensors
X_train_tensor = torch.tensor(X_train_bert, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_bert, dtype=torch.float32)
y_train_tensor = torch.tensor(train_labels_encoded, dtype=torch.long)
y_test_tensor = torch.tensor(test_labels_encoded, dtype=torch.long)

# Define dataset and dataloader
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Helps prevent overfitting
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Initialize model, optimizer, and loss function
num_classes = len(label_encoder.classes_)  # Get number of classes dynamically
model = MLPClassifier(input_dim=768, num_classes=num_classes)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

# Train MLP Model
num_epochs = 0  # You can adjust based on performance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("\n🚀 Training MLP Classifier on BERT Embeddings...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    train_acc = correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {total_loss:.4f}, Train Accuracy: {train_acc:.4f}")

# Evaluate Model on Test Set
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

# Compute final accuracy
mlp_accuracy = accuracy_score(all_labels, all_preds)
print(f"\n✅ MLP Classifier (BERT Embeddings) Accuracy: {mlp_accuracy:.4f}")
