import os
import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv("data/go_emotions_dataset.csv")  # Update with the correct path

# Display first few rows
print(df.head())

# Remove unnecessary columns
df = df.drop(columns=["id", "example_very_unclear"])

# Get emotion columns (multi-label)
emotion_columns = df.columns[1:]

# Convert labels to lists of 0s and 1s (multi-label format)
df[emotion_columns] = df[emotion_columns].astype(int)
df["labels"] = df[emotion_columns].values.tolist()  # Store multi-label format

# Keep only text and labels (REMOVE "label" since it does NOT exist)
df = df[["text", "labels"]]

# Show updated dataset
print(df.head())  # Verify labels are lists like [0,1,0,0,...]

# For initial testing, take a smaller sample
df_sample = df.sample(n=10000, random_state=42)  

from sklearn.model_selection import train_test_split

###########################################################################################
# Pre-Cleaning
###########################################################################################
import preprocessor
import contractions
import re
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

slang_dict = {"brb": "be right back", "idk": "I don't know", "u": "you"}

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", 
                       "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", 
                       "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", 
                       "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am",
                       "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", 
                       "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                       "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not",
                       "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", 
                       "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                       "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", 
                       "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                       "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is",
                       "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would",
                       "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have",
                       "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                       "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", 
                       "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
                       "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", 
                       "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", 
                       "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                       "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have",
                       "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'u.s':'america', 'e.g':'for example'}

punct = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-",
                 "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 
                 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', '!':' '}

mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater',
                'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ',
                'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can',
                'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 
                'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 
                'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 
                'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization',
                'demonetisation': 'demonetization'}

import emoji
from bs4 import BeautifulSoup
import string

def clean_text(text):
    '''Clean emoji, Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = emoji.demojize(text)
    text = re.sub(r'\:(.*?)\:','',text)
    text = str(text).lower()    #Making Text Lowercase
    text = re.sub('\[.*?\]', '', text)
    #The next 2 lines remove html text
    text = BeautifulSoup(text, 'lxml').get_text()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",", "'")
    text = re.sub(r"[^a-zA-Z?.!,¿']+", " ", text)
    return text

def clean_contractions(text, mapping):
    '''Clean contraction using contraction mapping'''    
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    for word in mapping.keys():
        if ""+word+"" in text:
            text = text.replace(""+word+"", ""+mapping[word]+"")
    #Remove Punctuations
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    return text

def clean_special_chars(text, punct, mapping):
    '''Cleans special characters present(if any)'''   
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text

def correct_spelling(x, dic):
    '''Corrects common spelling errors'''   
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x

def remove_space(text):
    '''Removes awkward spaces'''   
    #Removes awkward spaces 
    text = text.strip()
    text = text.split()
    return " ".join(text)

def text_preprocessing_pipeline(text):
    '''Cleaning and parsing the text.'''
    text = clean_text(text)
    text = clean_contractions(text, contraction_mapping)
    text = clean_special_chars(text, punct, punct_mapping)
    text = correct_spelling(text, mispell_dict)
    text = remove_space(text)
    return text

# Apply text cleaning to dataset
df_sample["cleaned_text"] = df_sample["text"].apply(text_preprocessing_pipeline)

# Display first few rows to verify
print(df_sample[["text", "cleaned_text"]].head())

# Use smaller dataset for training
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df_sample["cleaned_text"], df_sample["labels"], test_size=0.2, random_state=42
)

# Verify dataset shapes
print(f"Train samples: {len(train_texts)}, Test samples: {len(test_texts)}")

# Convert to list format
train_texts = train_texts.tolist()
test_texts = test_texts.tolist()
train_labels = np.array(train_labels.tolist(), dtype=np.float32)  # Convert labels to arrays
test_labels = np.array(test_labels.tolist(), dtype=np.float32)

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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

def run_logistic_regression_tfidf(X_train_tfidf, X_test_tfidf, train_labels, test_labels):

    print("------------------ Logistic Regression + TFIDF ------------------")
    # Use OneVsRestClassifier for Multi-Label Classification
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))

    # Train model
    clf.fit(X_train_tfidf, train_labels)

    # Predict on test set
    y_pred = clf.predict(X_test_tfidf)

    # Evaluate multi-label accuracy using F1-Score (Better for Multi-Label)
    f1_micro = f1_score(test_labels, y_pred, average="micro")  # Micro F1-Score
    f1_macro = f1_score(test_labels, y_pred, average="macro")  # Macro F1-Score

    print(f"\n Logistic Regression Multi-Label F1 Score (Micro): {f1_micro:.2f}")
    print(f"\n Logistic Regression Multi-Label F1 Score (Macro): {f1_macro:.2f}")

###########################################################################################
# SVM
###########################################################################################

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

def run_svm_tfidf(X_train_tfidf, X_test_tfidf, train_labels, test_labels):

    print("------------------ SVM + TFIDF ------------------")

    # Define all kernels to test
    kernels = ["linear", "poly", "rbf", "sigmoid"]

    # Loop through kernels
    for kernel in kernels:
        print(f"\n Training Multi-Label SVM with {kernel} kernel...")

        # Use OneVsRestClassifier for Multi-Label
        svm_clf = OneVsRestClassifier(SVC(kernel=kernel, degree=3, gamma="scale"))

        # Train SVM Model
        svm_clf.fit(X_train_tfidf, train_labels)

        # Predict on test set
        y_pred_svm = svm_clf.predict(X_test_tfidf)

        # Evaluate Multi-Label F1-Score
        f1_micro_svm = f1_score(test_labels, y_pred_svm, average="micro")
        f1_macro_svm = f1_score(test_labels, y_pred_svm, average="macro")

        print(f"\n Multi-Label SVM ({kernel}) F1 Score (Micro): {f1_micro_svm:.2f}")
        print(f" Multi-Label SVM ({kernel}) F1 Score (Macro): {f1_macro_svm:.2f}")

###########################################################################################
# Using BERT to expand text instead of TF-IDF
###########################################################################################
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import numpy as np

from transformers import AutoTokenizer, AutoModel

from huggingface_hub import login


# Load pre-trained BERT tokenizer & model - Baseline BERT model 
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# bert_model = BertModel.from_pråetrained("bert-base-uncased")

num_labels = len(emotion_columns)
print(f"num_labels: {num_labels}")

from transformers import BertForSequenceClassification, BertTokenizer
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_bert_emotions")
# bert_model = BertForSequenceClassification.from_pretrained("fine_tuned_bert_emotions", num_labels=num_labels)

# Load BERT model with sigmoid activation for multi-label classification
bert_model = BertForSequenceClassification.from_pretrained(
    "fine_tuned_bert_emotions", 
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


import torch

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
                batch_texts = text_list[i : i + batch_size]

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
X_train_bert = get_bert_embeddings(train_texts, batch_size=32)
X_test_bert = get_bert_embeddings(test_texts, batch_size=32)

# ###########################################################################################
# # BERT + Logistic Regression
# ###########################################################################################

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def run_logistic_regression_bert(X_train_bert, X_test_bert, train_labels, test_labels):

    print("------------------ Logistic Regression + BERT ------------------")

    # Convert multi-label one-hot encoding to single-class labels (Multiclass)
    train_labels_multiclass = np.argmax(train_labels, axis=1)  # Convert to class indices
    test_labels_multiclass = np.argmax(test_labels, axis=1)

    # Train Logistic Regression model for Multiclass classification
    clf_bert = LogisticRegression(max_iter=1000, solver="lbfgs")  
    clf_bert.fit(X_train_bert, train_labels_multiclass)

    # Predict on test set
    y_pred_bert = clf_bert.predict(X_test_bert)

    # Evaluate accuracy & F1-score for multiclass classification
    accuracy_bert = accuracy_score(test_labels_multiclass, y_pred_bert)
    f1_micro = f1_score(test_labels_multiclass, y_pred_bert, average="micro")
    f1_macro = f1_score(test_labels_multiclass, y_pred_bert, average="macro")

    print(f"\nLogistic Regression (BERT) Accuracy: {accuracy_bert:.2f}")
    print(f"Logistic Regression (BERT) F1 Score (Micro): {f1_micro:.2f}")
    print(f"Logistic Regression (BERT) F1 Score (Macro): {f1_macro:.2f}")

###########################################################################################
# SVM + BERT
###########################################################################################

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def run_svm_bert(X_train_bert, X_test_bert, train_labels, test_labels):

    print("------------------ SVM + BERT ------------------")

    # Try different kernels
    kernels = ["linear", "poly", "rbf", "sigmoid"]

    for kernel in kernels:
        print(f"\nTraining BERT + SVM with {kernel} kernel...")
        # svm_clf = SVC(kernel=kernel, degree=3, gamma="scale")  # Set degree for poly
        # Wrap SVC in OneVsRestClassifier for multi-label classification
        svm_clf = OneVsRestClassifier(SVC(kernel=kernel, degree=3, gamma="scale", probability=True))
        svm_clf.fit(X_train_bert, train_labels)

        # Predict on test set
        y_pred_svm = svm_clf.predict(X_test_bert)

        # Evaluate using F1-Score (better for multi-label classification)
        f1_micro_svm = f1_score(test_labels, y_pred_svm, average="micro")
        f1_macro_svm = f1_score(test_labels, y_pred_svm, average="macro")

        print(f"\nMulti-Label SVM ({kernel}) F1 Score (Micro): {f1_micro_svm:.2f}")
        print(f"Multi-Label SVM ({kernel}) F1 Score (Macro): {f1_macro_svm:.2f}")


###########################################################################################
# MLP + BERT (Multiclass)
###########################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

# Define MLP Classifier for Multiclass
class MLPClassifier(nn.Module):
    # def __init__(self, input_dim, num_classes):
    #     super(MLPClassifier, self).__init__()
    #     # self.fc1 = nn.Linear(input_dim, 256)
    #     self.fc1 = nn.Linear(input_dim, 512)
    #     self.norm1 = nn.LayerNorm(512)  # Layer Normalization
    #     self.relu1 = nn.ReLU()
    #     self.dropout1 = nn.Dropout(0.3)  # prevent overfitting
        
    #     # self.fc2 = nn.Linear(256, num_classes)
    #     self.fc2 = nn.Linear(512, 256)  # Additional hidden layer
    #     self.norm2 = nn.LayerNorm(256)  # Layer Normalization
    #     self.relu2 = nn.ReLU()
    #     self.dropout2 = nn.Dropout(0.3)

    #     self.fc3 = nn.Linear(256, num_classes)  # Output layer
    
    # def forward(self, x):
    #     x = self.fc1(x)
    #     x = self.norm1(x)       # Apply LayerNorm after linear layer
    #     x = self.relu1(x)
    #     x = self.dropout1(x)

    #     x = self.fc2(x)
    #     x = self.norm2(x)       # Apply LayerNorm again
    #     x = self.relu2(x)
    #     x = self.dropout2(x)

    #     x = self.fc3(x)         # Output logits (no softmax/sigmoid here)

    #     return x
    def __init__(self, input_dim, layer_dims, num_classes):
        super(MLPClassifier, self).__init__()

        layers = []
        current_dim = input_dim
        for dim in layer_dims:
            layers.append(nn.Linear(current_dim, dim))
            layers.append(nn.BatchNorm2d(dim))  # USE BATCH NORM 1 or 2D check both
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3)) # DOn't use drop out originally, unless I get good enough results
            current_dim = dim

        layers.append(nn.Linear(current_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def prepare_mlp_dataloader(X_train_bert, X_test_bert, train_labels, test_labels, batch_size=64):

    # Convert labels to numerical format for Multiclass Classification
    X_train_tensor = torch.tensor(X_train_bert, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_bert, dtype=torch.float32)
    y_train_tensor = torch.tensor(train_labels, dtype=torch.float32)  # float for BCE
    y_test_tensor = torch.tensor(test_labels, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_mlp_model(model, train_loader, test_loader, device, num_epochs, save_path):
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    # make LR higher, 
    loss_fn = nn.BCEWithLogitsLoss()  # Multi-label loss

    # Move model to GPU if available
    model.to(device)

    # For saving optimal epoch
    best_f1 = 0.0

    # For visualization of progress over epochs
    history = {
        "train_loss": [],
        "train_f1_micro": [],
        "test_f1_micro": [],
    }

    # Train MLP Model
    print("\n Training MLP Classifier on BERT Embeddings...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        y_true_train, y_pred_train = [], []
 
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int().cpu().numpy()
            y_pred_train.extend(preds)
            y_true_train.extend(y_batch.cpu().numpy())
        
        train_f1_micro = f1_score(y_true_train, y_pred_train, average="micro")
        history["train_loss"].append(total_loss)
        history["train_f1_micro"].append(train_f1_micro)

                # Eval on test
        model.eval()
        y_true_test, y_pred_test = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int().cpu().numpy()
                y_pred_test.extend(preds)
                y_true_test.extend(y_batch.cpu().numpy())
        test_f1_micro = f1_score(y_true_test, y_pred_test, average="micro")
        history["test_f1_micro"].append(test_f1_micro)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f} | Train F1 (Micro): {train_f1_micro:.4f} | Test F1 (Micro): {test_f1_micro:.4f}")

        # Save best model
        if test_f1_micro > best_f1:
            best_f1 = test_f1_micro
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved (Test F1 Micro: {test_f1_micro:.4f})")

    return history

import matplotlib.pyplot as plt

def plot_training_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, history["train_f1_micro"], label="Train F1 (Micro)")
    plt.plot(epochs, history["test_f1_micro"], label="Test F1 (Micro)")
    
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title("MLP Training Performance Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def evaluate_mlp_model(model, train_loader, test_loader, device):
    # Evaluate Model on Test Set
    def _evaluate(loader, split_name):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                probs = torch.sigmoid(logits)  # Convert to probabilities
                preds = (probs > 0.5).int().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.cpu().numpy())

        # Compute final accuracy & F1-Score for multiclass classification
        mlp_accuracy = accuracy_score(all_labels, all_preds)
        mlp_f1_micro = f1_score(all_labels, all_preds, average="micro")
        mlp_f1_macro = f1_score(all_labels, all_preds, average="macro")

        print(f"\n{split_name} Set Metrics for MLP + BERT:")
        print(f"\nOverall Accuracy: {mlp_accuracy:.4f}")
        print(f"Overall F1 Score (Micro): {mlp_f1_micro:.4f}")
        print(f"Overall F1 Score (Macro): {mlp_f1_macro:.4f}")

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        # Per-class precision, recall, F1
        precisions, recalls, f1s, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        emotion_labels = emotion_columns.tolist()

        print("\nPer-class Precision / Recall / F1:")
        for idx, label in enumerate(emotion_labels):  # or custom label list
            print(f"{label:20s} | P: {precisions[idx]:.2f} | R: {recalls[idx]:.2f} | F1: {f1s[idx]:.2f}")
        
        return mlp_f1_micro, mlp_f1_macro

    # Evaluate both train and test
    mlp_f1_micro_train, mlp_f1_macro_train = _evaluate(train_loader, "Train")
    mlp_f1_micro_test, mlp_f1_macro_test = _evaluate(test_loader, "Test")

    return mlp_f1_micro_train, mlp_f1_macro_train, mlp_f1_micro_test, mlp_f1_macro_test


def run_mlp_bert(X_train_bert, X_test_bert, train_labels, test_labels, layer_dims, label, num_epochs):
    print(f"------------------ MLP + BERT: {label} ------------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = prepare_mlp_dataloader(
        X_train_bert, X_test_bert, train_labels, test_labels
    )

    num_classes = train_labels.shape[1]
    model = MLPClassifier(input_dim=768, layer_dims=layer_dims, num_classes=num_classes)

    save_path = f"data/best_mlp_model_{label}.pt"

    # Load best model for final evaluation
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print("\nLoaded best model from disk for final evaluation for {label}.")


    history = train_mlp_model(model, train_loader, test_loader, device, num_epochs, save_path)
    mlp_f1_micro_train, mlp_f1_macro_train, mlp_f1_micro_test, mlp_f1_macro_test = evaluate_mlp_model(model, train_loader, test_loader, device)
    plot_training_history(history)

    return mlp_f1_micro_train, mlp_f1_macro_train, mlp_f1_micro_test, mlp_f1_macro_test

# ###########################################################################################
# # CNN + BERT 
# ###########################################################################################

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
# Controller to run selected models
###########################################################################################

def run_selected_models(models_to_run):

    print(f"Running training using the following models: {models_to_run}")

    mlp_results = []

    num_epochs = 2000

    if "logreg_tfidf" in models_to_run:
        run_logistic_regression_tfidf(X_train_tfidf, X_test_tfidf, train_labels, test_labels)

    if "svm_tfidf" in models_to_run:
        run_svm_tfidf(X_train_tfidf, X_test_tfidf, train_labels, test_labels)

    if "logreg_bert" in models_to_run:
        run_logistic_regression_bert(X_train_bert, X_test_bert, train_labels, test_labels)

    if "svm_bert" in models_to_run:
        run_svm_bert(X_train_bert, X_test_bert, train_labels, test_labels)

    if "mlp_bert" in models_to_run:
        # run_mlp_bert(X_train_bert, X_test_bert, train_labels, test_labels, num_epochs)
        
        # Run 2-layer MLP (default)
        f1_2 = run_mlp_bert(X_train_bert, X_test_bert, train_labels, test_labels,
                     layer_dims=[512, 256], label="2layer", num_epochs=num_epochs)
        
        # change to bigger dimensions then go to smaller dimensions
        # go to smaller number before classification

        # Run 3-layer MLP (deeper network)
        f1_3 = run_mlp_bert(X_train_bert, X_test_bert, train_labels, test_labels,
                     layer_dims=[768, 512, 256], label="3layer", num_epochs=num_epochs)
        
        mlp_results.append(("MLP 2-layer", *f1_2))
        mlp_results.append(("MLP 3-layer", *f1_3))

        # Print comparison
        print("\n================ Final Model Comparison (MLP + BERT) ================\n")
        print(f"{'Model':<15} | {'Train Micro':>11} | {'Train Macro':>11} | {'Test Micro':>10} | {'Test Macro':>10}")
        print("-" * 65)
        for label, f1_micro_train, f1_macro_train, f1_micro_test, f1_macro_test in mlp_results:
            print(f"{label:<15} | {f1_micro_train:11.4f} | {f1_macro_train:11.4f} | {f1_micro_test:10.4f} | {f1_macro_test:10.4f}")

###########################################################################################
# Run Models
###########################################################################################

models_to_run = [
    # "logreg_tfidf",
    # "svm_tfidf",
    # "logreg_bert",
    # "svm_bert",
    "mlp_bert"
]

run_selected_models(models_to_run)

