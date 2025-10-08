import pandas as pd
from sklearn.model_selection import train_test_split

# Load Dataset
df = pd.read_csv("labeled_triggers_exploded.csv")

# Drop rows where no trigger was found
df = df[df["trigger_label"] != "No Trigger Found"]

# Display first few rows
print(df.head())

# # Text processing
# X = df["text"]
# y = df["trigger_label"]


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

    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    text = " ".join([slang_dict[word] if word in slang_dict else word for word in text.split()])

    # Makes BERT worse, but without BERT better
    #   text = " ".join([word for word in text.split() if word not in stop_words]) # Remove stopwords

    return text


# Apply text cleaning to dataset
df["cleaned_text"] = df["text"].apply(clean_text)

# Display first few rows to verify
print(df[["text", "cleaned_text"]].head())

# Use smaller dataset for training
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["cleaned_text"], df["trigger_label"], test_size=0.2, random_state=42
)

print(f"Train Samples: {len(train_texts)} | Test Samples: {len(test_texts)}")

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
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# Load pre-trained BERT tokenizer & model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")


# Function to tokenize text and extract embeddings with a more detailed progress bar
def get_bert_embeddings(text_list, batch_size=32):
    """Tokenize text and extract BERT embeddings with a detailed progress bar."""
    embeddings = []
    total_batches = len(text_list) // batch_size + int(len(text_list) % batch_size != 0)

    with tqdm(total=len(text_list), desc="Processing BERT Embeddings", unit="sentence") as pbar:
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i: i + batch_size]

            # Tokenize batch
            tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

            with torch.no_grad():
                outputs = bert_model(**tokens)

            # # Mean pooling: Average over all token embeddings
            # batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            # Max Pooling (Alternative to Mean Pooling)
            batch_embeddings = torch.max(outputs.last_hidden_state, dim=1)[0].cpu().numpy()
            embeddings.append(batch_embeddings)

            # Update progress bar by number of processed sentences
            pbar.update(len(batch_texts))

    return np.vstack(embeddings)  # Combine all batches


# Convert training and test texts into BERT embeddings
print(f"\nConverting training and test sets into BERT embeddings...")
X_train_bert = get_bert_embeddings(train_texts.tolist(), batch_size=32)
X_test_bert = get_bert_embeddings(test_texts.tolist(), batch_size=32)

print("BERT embeddings shape:", X_train_bert.shape)  # Should be (num_samples, 768)

print(f"Total Labels: {len(train_labels)}")
print(f"Total BERT embeddings: {X_train_bert.shape[0]}")
print(f"Missing Labels: {df['trigger_label'].isnull().sum()}")
print(f"Unique Labels: {df['trigger_label'].nunique()}")
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

# ###########################################################################################
# # BERT + SVM Approximations to view progress
# ###########################################################################################
# from sklearn.linear_model import SGDClassifier
# from sklearn.kernel_approximation import RBFSampler, PolynomialCountSketch
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score
# from sklearn.utils import shuffle

# # Ensure tqdm works inside pandas apply() calls
# tqdm.pandas()

# # Function to display progress during feature transformation
# def transform_with_progress(transformer, X, description="Transforming"):
#     """Apply a transformation with a progress bar."""
#     with tqdm(total=X.shape[0], desc=description, unit="samples") as pbar:
#         X_transformed = transformer.fit_transform(X)
#         pbar.update(X.shape[0])  # Mark all samples as processed
#     return X_transformed

# # Define different loss functions for SGDClassifier
# loss_functions = {
#     "Linear SVM": SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3),
#     "Polynomial SVM": Pipeline([
#         ("poly_features", PolynomialCountSketch(degree=3, random_state=42)),
#         ("sgd", SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3))
#     ]),
#     "RBF SVM": Pipeline([
#         ("rbf_features", RBFSampler(gamma=0.1, random_state=42)),
#         ("sgd", SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3))
#     ]),
#     "Sigmoid SVM": SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3)
# }

# # Simulated BERT embeddings (Use actual BERT embeddings in practice)
# X_train_bert = np.random.rand(8000, 768)  # 8000 samples, 768 features
# X_test_bert = np.random.rand(2000, 768)   # 2000 samples, 768 features
# train_labels_encoded = np.random.randint(0, 28, 8000)  # Random labels for simulation
# test_labels_encoded = np.random.randint(0, 28, 2000)

# # Train and evaluate each model with progress bars
# for name, model in loss_functions.items():
#     print(f"\nTraining {name}...")

#     # Apply feature transformation with progress bar if needed
#     if isinstance(model, Pipeline):
#         X_train_transformed = transform_with_progress(model.named_steps[list(model.named_steps.keys())[0]], X_train_bert, description=f"Transforming Train Data ({name})")
#         X_test_transformed = transform_with_progress(model.named_steps[list(model.named_steps.keys())[0]], X_test_bert, description=f"Transforming Test Data ({name})")
#         model.named_steps["sgd"].fit(X_train_transformed, train_labels_encoded)
#     else:
#         model.fit(X_train_bert, train_labels_encoded)

#     # Training progress bar
#     with tqdm(total=100, desc=f"Training {name}", unit="%") as pbar:
#         for _ in range(100):
#             pbar.update(1)  # Simulated update for training

#     # Predict on test set
#     if isinstance(model, Pipeline):
#         y_pred_sgd = model.named_steps["sgd"].predict(X_test_transformed)
#     else:
#         y_pred_sgd = model.predict(X_test_bert)

#     # Evaluate accuracy
#     accuracy_sgd = accuracy_score(test_labels_encoded, y_pred_sgd)
#     print(f"{name} Accuracy: {accuracy_sgd:.2f}")

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
# Checking with Synonyms
###########################################################################################
import random
from nltk.corpus import wordnet


def get_synonyms(word):
    """Fetch synonyms from WordNet"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))  # Convert underscores to spaces
    return list(synonyms) if synonyms else []  # Return empty list if no synonyms found


def replace_with_synonyms(text):
    """Replace words in text with synonyms from dictionary"""
    words = text.split()
    new_words = []

    for word in words:
        synonyms = get_synonyms(word)  # Get synonyms for the word
        if synonyms:  # If synonyms exist, replace with a random one
            new_words.append(random.choice(synonyms))
        else:  # If no synonyms, keep the original word
            new_words.append(word)

    return " ".join(new_words)


# Apply to dataset
df["synonym_text"] = df["cleaned_text"].apply(replace_with_synonyms)

# Compare Original vs. Synonym-Based Text
print(df[["cleaned_text", "synonym_text"]].head())

# Transform synonym-based text into TF-IDF vectors
X_synonym_tfidf = vectorizer.transform(df["synonym_text"])

# Predict on Synonym-Text
y_pred_synonym = clf.predict(X_synonym_tfidf)

X_synonym_bert = get_bert_embeddings(df["synonym_text"].tolist(), batch_size=32)

# Predict on BERT Embeddings
y_pred_synonym_bert = clf_bert.predict(X_synonym_bert)

# Compare Original vs Synonym-Based Predictions
df["original_prediction"] = clf.predict(X_train_tfidf)
df["synonym_prediction"] = clf.predict(X_synonym_tfidf)

# Check how many predictions remain the same
df["prediction_match"] = df["original_prediction"] == df["synonym_prediction"]

# Measure robustness
robustness_score = df["prediction_match"].mean()
print(f"Robustness Score (How Often Predictions Stay the Same): {robustness_score:.2f}")
