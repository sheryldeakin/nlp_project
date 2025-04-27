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

    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X_train_tfidf, train_labels)
    y_pred = clf.predict(X_test_tfidf)

    # Evaluate
    f1_micro = f1_score(test_labels, y_pred, average="micro")
    f1_macro = f1_score(test_labels, y_pred, average="macro")
    accuracy = accuracy_score(test_labels, y_pred)

    history = {
        "train_loss": [0],
        "train_f1_micro": [f1_micro],  # Placeholder for compatibility
        "test_f1_micro": [f1_micro],
        "train_accuracy": [accuracy],  # Placeholder
        "test_accuracy": [accuracy],
    }

    best_epoch_test_accuracy = np.argmax(history["test_accuracy"])
    best_accuracy_f1_micro_test_accuracy = history["test_f1_micro"][best_epoch_test_accuracy]
    best_accuracy_test_accuracy = history["test_accuracy"][best_epoch_test_accuracy]

    best_epoch_f1_micro = np.argmax(history["test_f1_micro"])
    best_f1_micro_test_accuracy = history["test_f1_micro"][best_epoch_f1_micro]
    best_f1_test_accuracy = history["test_accuracy"][best_epoch_f1_micro]   

    print(f"LogReg + TFIDF | Acc: {accuracy:.4f} | F1 Micro: {f1_micro:.4f} | F1 Macro: {f1_macro:.4f}")
    return history, f1_micro, f1_macro, f1_micro, f1_macro, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy



###########################################################################################
# SVM
###########################################################################################

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

def run_svm_tfidf(X_train_tfidf, X_test_tfidf, train_labels, test_labels):
    print("------------------ SVM + TFIDF ------------------")
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    results = {}

    for kernel in kernels:
        print(f"\nTraining SVM with {kernel} kernel...")
        clf = OneVsRestClassifier(SVC(kernel=kernel, degree=3, gamma="scale", probability=True))
        clf.fit(X_train_tfidf, train_labels)
        y_pred = clf.predict(X_test_tfidf)

        f1_micro = f1_score(test_labels, y_pred, average="micro")
        f1_macro = f1_score(test_labels, y_pred, average="macro")
        accuracy = accuracy_score(test_labels, y_pred)

        label = f"SVM TFIDF ({kernel})"
        history = {
            "train_loss": [0],
            "train_f1_micro": [f1_micro],
            "test_f1_micro": [f1_micro],
            "train_accuracy": [accuracy],
            "test_accuracy": [accuracy],
        }


        best_epoch_test_accuracy = np.argmax(history["test_accuracy"])
        best_accuracy_f1_micro_test_accuracy = history["test_f1_micro"][best_epoch_test_accuracy]
        best_accuracy_test_accuracy = history["test_accuracy"][best_epoch_test_accuracy]

        best_epoch_f1_micro = np.argmax(history["test_f1_micro"])
        best_f1_micro_test_accuracy = history["test_f1_micro"][best_epoch_f1_micro]
        best_f1_test_accuracy = history["test_accuracy"][best_epoch_f1_micro]   

        print(f"{label} | Acc: {accuracy:.4f} | F1 Micro: {f1_micro:.4f} | F1 Macro: {f1_macro:.4f}")
        results[label] = (history, f1_micro, f1_macro, f1_micro, f1_macro, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy)

    return results   # dict of kernel: (history, ...)


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

# ###########################################################################################
# # BERT + Logistic Regression
# ###########################################################################################

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def run_logistic_regression_bert(X_train_bert, X_test_bert, train_labels, test_labels):
    print("------------------ Logistic Regression + BERT ------------------")

    # === reshape to 2D via mean pooling ===
    X_train_flat = X_train_bert.mean(axis=1)  # shape: [num_samples, hidden_dim]
    X_test_flat = X_test_bert.mean(axis=1)

    train_labels_multiclass = np.argmax(train_labels, axis=1)
    test_labels_multiclass = np.argmax(test_labels, axis=1)


    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_flat, train_labels_multiclass)
    y_pred = clf.predict(X_test_flat)

    f1_micro = f1_score(test_labels_multiclass, y_pred, average="micro")
    f1_macro = f1_score(test_labels_multiclass, y_pred, average="macro")
    accuracy = accuracy_score(test_labels_multiclass, y_pred)

    history = {
        "train_loss": [0],
        "train_f1_micro": [f1_micro],
        "test_f1_micro": [f1_micro],
        "train_accuracy": [accuracy],
        "test_accuracy": [accuracy],
    }

    best_epoch_test_accuracy = np.argmax(history["test_accuracy"])
    best_accuracy_f1_micro_test_accuracy = history["test_f1_micro"][best_epoch_test_accuracy]
    best_accuracy_test_accuracy = history["test_accuracy"][best_epoch_test_accuracy]

    best_epoch_f1_micro = np.argmax(history["test_f1_micro"])
    best_f1_micro_test_accuracy = history["test_f1_micro"][best_epoch_f1_micro]
    best_f1_test_accuracy = history["test_accuracy"][best_epoch_f1_micro]   
 
    print(f"LogReg + BERT | Acc: {accuracy:.4f} | F1 Micro: {f1_micro:.4f} | F1 Macro: {f1_macro:.4f}")
    return history, f1_micro, f1_macro, f1_micro, f1_macro, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy


###########################################################################################
# SVM + BERT
###########################################################################################

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def run_svm_bert(X_train_bert, X_test_bert, train_labels, test_labels):
    print("------------------ SVM + BERT ------------------")
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    results = {}

    X_train_flat = X_train_bert.mean(axis=1)
    X_test_flat = X_test_bert.mean(axis=1)


    for kernel in kernels:
        print(f"\nTraining SVM + BERT with {kernel} kernel...")
        clf = OneVsRestClassifier(SVC(kernel=kernel, degree=3, gamma="scale", probability=True))
        clf.fit(X_train_flat, train_labels)
        y_pred = clf.predict(X_test_flat)

        f1_micro = f1_score(test_labels, y_pred, average="micro")
        f1_macro = f1_score(test_labels, y_pred, average="macro")
        accuracy = accuracy_score(test_labels, y_pred)

        label = f"SVM BERT ({kernel})"
        history = {
            "train_loss": [0],
            "train_f1_micro": [f1_micro],
            "test_f1_micro": [f1_micro],
            "train_accuracy": [accuracy],
            "test_accuracy": [accuracy],
        }

        best_epoch_test_accuracy = np.argmax(history["test_accuracy"])
        best_accuracy_f1_micro_test_accuracy = history["test_f1_micro"][best_epoch_test_accuracy]
        best_accuracy_test_accuracy = history["test_accuracy"][best_epoch_test_accuracy]

        best_epoch_f1_micro = np.argmax(history["test_f1_micro"])
        best_f1_micro_test_accuracy = history["test_f1_micro"][best_epoch_f1_micro]
        best_f1_test_accuracy = history["test_accuracy"][best_epoch_f1_micro]  

        print(f"{label} | Acc: {accuracy:.4f} | F1 Micro: {f1_micro:.4f} | F1 Macro: {f1_macro:.4f}")
        results[label] = (history, f1_micro, f1_macro, f1_micro, f1_macro, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy)

    return results


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

    def __init__(self, input_dim, layer_dims, num_classes):
        super(MLPClassifier, self).__init__()

        layers = []
        current_dim = input_dim
        for dim in layer_dims:
            layers.append(nn.Linear(current_dim, dim))
            layers.append(nn.BatchNorm1d(dim))  # USE BATCH NORM 1 or 2D check both
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
        "train_accuracy": [],
        "test_accuracy": [],
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
        train_accuracy = accuracy_score(np.array(y_true_train), np.array(y_pred_train))
        history["train_accuracy"].append(train_accuracy)

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
        test_accuracy = accuracy_score(np.array(y_true_test), np.array(y_pred_test))
        history["test_accuracy"].append(test_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f} | Train Acc: {train_accuracy:.4f} | Train F1: {train_f1_micro:.4f} | Test Acc: {test_accuracy:.4f} | Test F1: {test_f1_micro:.4f}")

        # # Save best model
        # if test_f1_micro > best_f1:
        #     best_f1 = test_f1_micro
        #     torch.save(model.state_dict(), save_path)
        #     print(f"New best model saved (Test F1 Micro: {test_f1_micro:.4f})")
        
        best_epoch_test_accuracy = np.argmax(history["test_accuracy"])
        best_accuracy_f1_micro_test_accuracy = history["test_f1_micro"][best_epoch_test_accuracy]
        best_accuracy_test_accuracy = history["test_accuracy"][best_epoch_test_accuracy]

        best_epoch_f1_micro = np.argmax(history["test_f1_micro"])
        best_f1_micro_test_accuracy = history["test_f1_micro"][best_epoch_f1_micro]
        best_f1_test_accuracy = history["test_accuracy"][best_epoch_f1_micro]  

    return history, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy

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

def evaluate_mlp_model(model, train_loader, test_loader, device, label_names):
    # Evaluate Model on Test Set
    def _evaluate(loader, split_name, label_names):
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

        if isinstance(label_names, (list, np.ndarray)):
            emotion_labels = label_names.tolist() if isinstance(label_names, np.ndarray) else label_names
        else:
            raise ValueError(f"Expected label_names to be a list or ndarray, but got: {type(label_names).__name__}, label_names: {label_names}")



        print("\nPer-class Precision / Recall / F1:")
        for idx, label in enumerate(emotion_labels):  # or custom label list
            print(f"{label:20s} | P: {precisions[idx]:.2f} | R: {recalls[idx]:.2f} | F1: {f1s[idx]:.2f}")
        
        return mlp_f1_micro, mlp_f1_macro

    # Evaluate both train and test
    mlp_f1_micro_train, mlp_f1_macro_train = _evaluate(train_loader, "Train", label_names)
    mlp_f1_micro_test, mlp_f1_macro_test = _evaluate(test_loader, "Test", label_names)

    return mlp_f1_micro_train, mlp_f1_macro_train, mlp_f1_micro_test, mlp_f1_macro_test


def run_mlp_bert(X_train_bert, X_test_bert, train_labels, test_labels, layer_dims, label, num_epochs, label_names):
    print(f"------------------ MLP + BERT: {label} ------------------")

    X_train_flat = X_train_bert.mean(axis=1)  # [num_samples, hidden_dim]
    X_test_flat = X_test_bert.mean(axis=1)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = prepare_mlp_dataloader(
        X_train_flat, X_test_flat, train_labels, test_labels
    )

    num_classes = train_labels.shape[1]
    model = MLPClassifier(input_dim=768, layer_dims=layer_dims, num_classes=num_classes)

    save_path = f"data/best_mlp_model_{label}.pt"

    # # Load best model for final evaluation
    # if os.path.exists(save_path):
    #     model.load_state_dict(torch.load(save_path))
    #     print("\nLoaded best model from disk for final evaluation for {label}.")


    history, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy = train_mlp_model(model, train_loader, test_loader, device, num_epochs, save_path)
    mlp_f1_micro_train, mlp_f1_macro_train, mlp_f1_micro_test, mlp_f1_macro_test = evaluate_mlp_model(model, train_loader, test_loader, device, label_names=label_names)

    return history, mlp_f1_micro_train, mlp_f1_macro_train, mlp_f1_micro_test, mlp_f1_macro_test, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy


# ###########################################################################################
# # CNN + BERT 
# ###########################################################################################
import torch.nn.functional as F

import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits (BCEWithLogits compatible)
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_term = (1 - pt) ** self.gamma

        loss = self.alpha * focal_term * BCE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss



class CNNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, conv_configs, dropout=0.3):
        """
        conv_configs: list of tuples -> (out_channels, kernel_size)
        """
        super(CNNClassifier, self).__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=input_dim, out_channels=out_ch, kernel_size=k),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)  # Get one value per filter
            )
            for out_ch, k in conv_configs
        ])

        total_out = sum([cfg[0] for cfg in conv_configs])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(total_out, num_classes)

    def forward(self, x):
        # x = x.unsqueeze(1).repeat(1, 768, 1)  # (B, input_dim, 1) to (B, input_dim, seq_len)
        # x = x.transpose(1, 2)  # (B, seq_len, input_dim) → (B, input_dim, seq_len)
        # x = [conv(x).squeeze(2) for conv in self.convs]  # Apply each conv block
        # x = torch.cat(x, dim=1)  # Concatenate all pooled features
        # return self.fc(x)
        x = x.permute(0, 2, 1)  # -> [batch_size, hidden_size, seq_len]
        conv_outs = [F.relu(conv(x)) for conv in self.convs]  # list of [batch, filters, ~]
        pooled = [F.max_pool1d(c, kernel_size=c.shape[2]).squeeze(2) for c in conv_outs]  # [batch, filters]
        out = torch.cat(pooled, dim=1)
        out = self.dropout(out)
        return self.fc(out)

def train_cnn_model(model, train_loader, test_loader, device, num_epochs, save_path, lr=1e-4, loss_type="bce", weights=0):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    best_f1 = 0.0

    model.to(device)

    if loss_type == "bce":
        loss_fn = nn.BCEWithLogitsLoss()
    elif loss_type == "weighted_bce":
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights.to(device))
    elif loss_type == "focal":
        loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
    else:
        raise ValueError("Invalid loss_type")

    history = {
        "train_loss": [],
        "train_f1_micro": [],
        "test_f1_micro": [],
        "train_accuracy": [],
        "test_accuracy": [],
    }

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

        train_f1 = f1_score(y_true_train, y_pred_train, average="micro")
        train_acc = accuracy_score(y_true_train, y_pred_train)
        history["train_loss"].append(total_loss)
        history["train_f1_micro"].append(train_f1)
        history["train_accuracy"].append(train_acc)

        # Evaluation
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

        test_f1 = f1_score(y_true_test, y_pred_test, average="micro")
        test_acc = accuracy_score(y_true_test, y_pred_test)
        history["test_f1_micro"].append(test_f1)
        history["test_accuracy"].append(test_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")

        # if test_f1 > best_f1:
        #     best_f1 = test_f1
        #     torch.save(model.state_dict(), save_path)
        #     print(f"New best CNN model saved for test F1: {test_f1:.4f}")

        best_epoch_test_accuracy = np.argmax(history["test_accuracy"])
        best_accuracy_f1_micro_test_accuracy = history["test_f1_micro"][best_epoch_test_accuracy]
        best_accuracy_test_accuracy = history["test_accuracy"][best_epoch_test_accuracy]

        best_epoch_f1_micro = np.argmax(history["test_f1_micro"])
        best_f1_micro_test_accuracy = history["test_f1_micro"][best_epoch_f1_micro]
        best_f1_test_accuracy = history["test_accuracy"][best_epoch_f1_micro] 

    return history, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy


import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

def evaluate_cnn_model(model, train_loader, test_loader, device, label="cnn_model", label_names=""):
    def _evaluate(loader, split_name, label_names):
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1_micro = f1_score(all_labels, all_preds, average="micro")
        f1_macro = f1_score(all_labels, all_preds, average="macro")

        print(f"\n{split_name} Set Metrics for CNN + BERT:")
        print(f"Overall Accuracy: {acc:.4f}")
        print(f"F1 Score (Micro): {f1_micro:.4f}")
        print(f"F1 Score (Macro): {f1_macro:.4f}")

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        # Per-class metrics
        precisions, recalls, f1s, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        if isinstance(label_names, (list, np.ndarray)):
            emotion_labels = label_names.tolist() if isinstance(label_names, np.ndarray) else label_names
        else:
            raise ValueError(f"Expected label_names to be a list or ndarray, but got: {type(label_names).__name__}, label_names: {label_names}")


        # Store per-class results
        df = pd.DataFrame({
            "Label": emotion_labels,
            "Precision": precisions,
            "Recall": recalls,
            "F1": f1s
        })

        # Save to CSV
        save_dir = "logs"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{label}_{split_name.lower()}_metrics.csv")
        df.to_csv(save_path, index=False)
        print(f"\nPer-class metrics saved to {save_path}")

        # Print final sorted values
        print(f"\n{split_name} Set — Top Emotions by F1:")
        print(df.sort_values(by="F1", ascending=False).to_string(index=False))

        return f1_micro, f1_macro

    f1_micro_train, f1_macro_train = _evaluate(train_loader, "Train", label_names)
    f1_micro_test, f1_macro_test = _evaluate(test_loader, "Test", label_names)

    return f1_micro_train, f1_macro_train, f1_micro_test, f1_macro_test



def run_cnn_bert(X_train_bert, X_test_bert, train_labels, test_labels, conv_configs, label, num_epochs, dropout=0.3, lr=1e-4, loss_type='bce', label_names=""):
    print(f"------------------ CNN + BERT: {label} ------------------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = prepare_mlp_dataloader(X_train_bert, X_test_bert, train_labels, test_labels)

    num_classes = train_labels.shape[1]
    model = CNNClassifier(input_dim=768, num_classes=num_classes, conv_configs=conv_configs, dropout=dropout)
    save_path = f"data/best_cnn_model_{label}.pt"

    # Compute class frequencies
    label_counts = np.sum(train_labels, axis=0)
    total = train_labels.shape[0]

    # Avoid divide-by-zero
    weights = total / (label_counts + 1e-6)
    weights = torch.tensor(weights, dtype=torch.float32)

    # if os.path.exists(save_path):
    #     model.load_state_dict(torch.load(save_path))
    #     print(f"\nLoaded best CNN model for {label}.")

    history, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy = train_cnn_model(model, train_loader, test_loader, device, num_epochs, save_path, lr=lr, loss_type=loss_type, weights=weights)
    f1_micro_train, f1_macro_train, f1_micro_test, f1_macro_test = evaluate_cnn_model(model, train_loader, test_loader, device, label=label, label_names=label_names)

    return history, f1_micro_train, f1_macro_train, f1_micro_test, f1_macro_test, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy

# ###########################################################################################
# # BiLSTM + BERT
# ###########################################################################################
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # bi-directional
        self.norm = nn.LayerNorm(hidden_dim * 2)  # Because BiLSTM doubles the hidden size


    def forward(self, x):
        # x: [batch_size, seq_len, input_dim] => typically [B, 64, 768]
        lstm_out, _ = self.lstm(x)
        pooled = torch.mean(lstm_out, dim=1)  # mean pooling across sequence
        pooled = self.norm(pooled)
        out = self.dropout(pooled)
        return self.fc(out)

def train_lstm_model(model, train_loader, test_loader, device, num_epochs, save_path, lr=1e-8, weights=1):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights.to(device))
    best_f1 = 0.0
    model.to(device)

    history = {
        "train_loss": [], "train_f1_micro": [], "test_f1_micro": [],
        "train_accuracy": [], "test_accuracy": []
    }

    for epoch in range(num_epochs):
        model.train()
        total_loss, y_true_train, y_pred_train = 0, [], []

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

        train_f1 = f1_score(y_true_train, y_pred_train, average="micro")
        train_acc = accuracy_score(y_true_train, y_pred_train)
        history["train_loss"].append(total_loss)
        history["train_f1_micro"].append(train_f1)
        history["train_accuracy"].append(train_acc)

        # Evaluation
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

        test_f1 = f1_score(y_true_test, y_pred_test, average="micro")
        test_acc = accuracy_score(y_true_test, y_pred_test)
        history["test_f1_micro"].append(test_f1)
        history["test_accuracy"].append(test_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")

        # if test_f1 > best_f1:
        #     best_f1 = test_f1
        #     torch.save(model.state_dict(), save_path)
        #     print(f"New best LSTM model saved for test F1: {test_f1:.4f}")

        best_epoch_test_accuracy = np.argmax(history["test_accuracy"])
        best_accuracy_f1_micro_test_accuracy = history["test_f1_micro"][best_epoch_test_accuracy]
        best_accuracy_test_accuracy = history["test_accuracy"][best_epoch_test_accuracy]

        best_epoch_f1_micro = np.argmax(history["test_f1_micro"])
        best_f1_micro_test_accuracy = history["test_f1_micro"][best_epoch_f1_micro]
        best_f1_test_accuracy = history["test_accuracy"][best_epoch_f1_micro] 

    return history, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy


def run_lstm_bert(X_train_bert, X_test_bert, train_labels, test_labels, hidden_dim, num_layers, dropout, lr, label, num_epochs=200, label_names=""):
    print(f"------------------ BiLSTM + BERT: {label} ------------------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = prepare_mlp_dataloader(X_train_bert, X_test_bert, train_labels, test_labels)

    num_classes = train_labels.shape[1]
    model = BiLSTMClassifier(
        input_dim=768,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
        )
    
    # Compute class frequencies
    label_counts = np.sum(train_labels, axis=0)
    total = train_labels.shape[0]

    # Avoid divide-by-zero
    weights = total / (label_counts + 1e-6)
    weights = torch.tensor(weights, dtype=torch.float32)

    save_path = f"data/best_lstm_model_{label}.pt"

    # if os.path.exists(save_path):
    #     model.load_state_dict(torch.load(save_path))
    #     print(f"\nLoaded best BiLSTM model for {label}.")

    history, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy = train_lstm_model(model, train_loader, test_loader, device, num_epochs, save_path, lr=lr, weights=weights)
    f1_micro_train, f1_macro_train, f1_micro_test, f1_macro_test = evaluate_cnn_model(model, train_loader, test_loader, device, label=label, label_names=label_names)

    return history, f1_micro_train, f1_macro_train, f1_micro_test, f1_macro_test, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy

# ###########################################################################################
# # Straight up BERT
# ###########################################################################################
def tokenize_for_bert(texts, tokenizer, max_length=64):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return encodings["input_ids"], encodings["attention_mask"]
from torch.utils.data import Dataset

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.input_ids, self.attention_mask = tokenize_for_bert(texts, tokenizer, max_length)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

def prepare_bert_dataloaders(train_texts, test_texts, train_labels, test_labels, tokenizer, batch_size=32):
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
    test_dataset = EmotionDataset(test_texts, test_labels, tokenizer)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name="sdeakin/fine_tuned_bert_emotions", num_classes=28, dropout=0.3):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("sdeakin/fine_tuned_bert_emotions")
        # self.bert = BertForSequenceClassification.from_pretrained(
        #     "fine_tuned_bert_emotions",
        #     num_labels=num_labels,
        #     problem_type="multi_label_classification"
        # )
        self.dropout = nn.Dropout(dropout)
        # self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)  # 768 -> 28
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)  # 768 → 28 classes

    # def forward(self, input_ids, attention_mask):
    #     return self.bert(input_ids=input_ids, attention_mask=attention_mask).logits

    # def forward(self, input_ids, attention_mask):
    #     outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    #     cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    #     x = self.dropout(cls_output)
    #     return self.classifier(x)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # shape: [batch, seq_len, hidden]
        pooled = torch.mean(sequence_output, dim=1)  # mean pooling across tokens
        x = self.dropout(pooled)
        return self.classifier(x)

    # def forward(self, input_ids, attention_mask):
    #     outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    #     cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    #     cls_output = self.dropout(cls_output)
    #     return self.classifier(cls_output)
    

    


def train_bert_finetune_model(model, train_loader, test_loader, device, num_epochs, save_path, lr=2e-5, weights=1):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights.to(device))
    model.to(device)

    best_f1 = 0
    history = {"train_loss": [], "train_f1_micro": [], "test_f1_micro": [], "train_accuracy": [], "test_accuracy": []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        y_true_train, y_pred_train = [], []

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
            optimizer.step()

            total_loss += loss.item()

            probs = torch.sigmoid(outputs.logits)
            preds = (probs > 0.5).int().cpu().numpy()
            y_pred_train.extend(preds)
            y_true_train.extend(labels.cpu().numpy())

        train_f1 = f1_score(y_true_train, y_pred_train, average="micro")
        train_acc = accuracy_score(y_true_train, y_pred_train)

        history["train_loss"].append(total_loss)
        history["train_f1_micro"].append(train_f1)
        history["train_accuracy"].append(train_acc)

        model.eval()
        y_true_test, y_pred_test = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(logits.logits)
                preds = (probs > 0.5).int().cpu().numpy()
                y_pred_test.extend(preds)
                y_true_test.extend(labels.cpu().numpy())


        test_f1 = f1_score(y_true_test, y_pred_test, average="micro")
        test_acc = accuracy_score(y_true_test, y_pred_test)
        history["test_f1_micro"].append(test_f1)
        history["test_accuracy"].append(test_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")

        # if test_f1 > best_f1:
        #     best_f1 = test_f1
        #     torch.save(model.state_dict(), save_path)
        #     print(f"New best BERT model saved for test F1: {test_f1:.4f}")

        best_epoch_test_accuracy = np.argmax(history["test_accuracy"])
        best_accuracy_f1_micro_test_accuracy = history["test_f1_micro"][best_epoch_test_accuracy]
        best_accuracy_test_accuracy = history["test_accuracy"][best_epoch_test_accuracy]

        best_epoch_f1_micro = np.argmax(history["test_f1_micro"])
        best_f1_micro_test_accuracy = history["test_f1_micro"][best_epoch_f1_micro]
        best_f1_test_accuracy = history["test_accuracy"][best_epoch_f1_micro]  

    return history, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy


def evaluate_bert_model(model, train_loader, test_loader, device, label="bert_finetune", label_names=""):
    def _evaluate(loader, split_name, label_names):
        model.eval()
        model.to(device)
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(outputs.logits)
                preds = (probs > 0.5).int().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1_micro = f1_score(all_labels, all_preds, average="micro")
        f1_macro = f1_score(all_labels, all_preds, average="macro")

        print(f"\n{split_name} Set Metrics for {label}:")
        print(f" Accuracy: {acc:.4f}")
        print(f" F1 Score (Micro): {f1_micro:.4f}")
        print(f" F1 Score (Macro): {f1_macro:.4f}")

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        precisions, recalls, f1s, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        if isinstance(label_names, (list, np.ndarray)):
            emotion_labels = label_names.tolist() if isinstance(label_names, np.ndarray) else label_names
        else:
            raise ValueError(f"Expected label_names to be a list or ndarray, but got: {type(label_names).__name__}, label_names: {label_names}")



        df = pd.DataFrame({
            "Label": emotion_labels,
            "Precision": precisions,
            "Recall": recalls,
            "F1": f1s
        })

        save_dir = "logs"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{label}_{split_name.lower()}_metrics.csv")
        df.to_csv(save_path, index=False)
        print(f"\nPer-class metrics saved to {save_path}")

        print(f"\n🔍 {split_name} Set — Top Emotions by F1:")
        print(df.sort_values(by="F1", ascending=False).to_string(index=False))

        return f1_micro, f1_macro, acc

    f1_micro_train, f1_macro_train, train_acc = _evaluate(train_loader, "Train", label_names)
    f1_micro_test, f1_macro_test, test_acc = _evaluate(test_loader, "Test", label_names)

    return f1_micro_train, f1_macro_train, f1_micro_test, f1_macro_test



def run_finetuned_bert_model(train_texts, test_texts, train_labels, test_labels, label, num_epochs=200, dropout=0.3, lr=2e-5, label_names=""):
    print(f"------------------ Fine-Tuned BERT: {label} ------------------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(
    "sdeakin/fine_tuned_bert_emotions",
    problem_type="multi_label_classification",
    ignore_mismatched_sizes=True
    )
    tokenizer = BertTokenizer.from_pretrained("sdeakin/fine_tuned_bert_emotions")



    train_loader, test_loader = prepare_bert_dataloaders(train_texts, test_texts, train_labels, test_labels, tokenizer)

    # Replace classifier dynamically
    model.classifier = torch.nn.Linear(model.config.hidden_size, train_labels.shape[1])
    model.num_labels = train_labels.shape[1]

    # model = BERTClassifier("fine_tuned_bert_emotions", num_classes=train_labels.shape[1], dropout=dropout)
    save_path = f"data/best_finetuned_bert_model_{label}.pt"

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print(f"\nLoaded best BERT model for {label}.")

    # Compute class frequencies
    label_counts = np.sum(train_labels, axis=0)
    total = train_labels.shape[0]

    # Avoid divide-by-zero
    weights = total / (label_counts + 1e-6)
    weights = torch.tensor(weights, dtype=torch.float32)

    history, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy = train_bert_finetune_model(model, train_loader, test_loader, device, num_epochs, save_path, lr=lr, weights=weights)

    f1_micro_train, f1_macro_train, f1_micro_test, f1_macro_test = evaluate_bert_model(
        model, train_loader, test_loader, device, label=label, label_names=label_names
    )   


    return history, f1_micro_train, f1_macro_train, f1_micro_test, f1_macro_test, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy

import colorcet as cc

###########################################################################################
# Controller to run selected models
###########################################################################################

import colorsys

def generate_distinct_colors(n):
    """Generate `n` visually distinct RGB colors."""
    hues = [i / n for i in range(n)]
    return [colorsys.hsv_to_rgb(h, 0.7, 0.9) for h in hues]


def plot_all_model_histories(history_dict, label_prefix):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Glasbey colormap for categorical distinction (up to 256 visually distinct colors)
    colors = cc.glasbey[:len(history_dict)]

    # colors = generate_distinct_colors(len(history_dict))

    lines = []
    labels = []

    for idx, (model_name, hist) in enumerate(history_dict.items()):
        color = colors[idx % len(colors)]

        line, = axs[0].plot(hist["test_f1_micro"], label=model_name, color=color)
        axs[1].plot(hist["test_accuracy"], color=color)
        axs[2].plot(hist["train_loss"], color=color)

        lines.append(line)
        labels.append(model_name)

    axs[0].set_title("Test F1 (Micro)")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("F1 Score")
    axs[0].grid()

    axs[1].set_title("Test Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].grid()

    axs[2].set_title("Train Loss")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Loss")
    axs[2].grid()

    # Layout & legend
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    fig.legend(lines, labels, loc="center left", bbox_to_anchor=(0.81, 0.5), fontsize="small")

    save_dir = f"plots/{label_prefix}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"all_models_training_plot.png"))
    plt.show()

import matplotlib.pyplot as plt
import os
import colorcet as cc
from collections import defaultdict

def plot_model_groups(history_dict, label_prefix):

    os.makedirs(f"plots/grouped/{label_prefix}", exist_ok=True)

    # Updated grouping logic
    categories = {
        "CNN": ["cnn"],
        "MLP": ["mlp"],
        "BiLSTM": ["lstm", "bilstm"],
        "BERT Finetuned": ["finetune", "bert finetune"],
        "LogReg": ["logreg"],
        "SVM": ["svm"],
    }

    grouped = defaultdict(dict)

    for model_name, hist in history_dict.items():
        model_name_lower = model_name.lower()
        matched = False
        for group, keywords in categories.items():
            if any(kw in model_name_lower for kw in keywords):
                grouped[group][model_name] = hist
                matched = True
                break
        if not matched:
            grouped["Other"][model_name] = hist

    for group_name, models in grouped.items():
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        colors = cc.glasbey[:len(models)]

        lines, labels = [], []

        for idx, (model_name, hist) in enumerate(models.items()):
            color = colors[idx]
            line, = axs[0].plot(hist["test_f1_micro"], label=model_name, color=color)
            axs[1].plot(hist["test_accuracy"], color=color)
            axs[2].plot(hist["train_loss"], color=color)

            lines.append(line)
            labels.append(model_name)

        axs[0].set_title(f"{group_name} - Test F1 (Micro)")
        axs[1].set_title(f"{group_name} - Test Accuracy")
        axs[2].set_title(f"{group_name} - Train Loss")

        for ax in axs:
            ax.set_xlabel("Epoch")
            ax.grid()

        plt.tight_layout(rect=[0, 0, 0.8, 1])
        fig.legend(lines, labels, loc="center left", bbox_to_anchor=(0.81, 0.5), fontsize="small")
        plt.savefig(f"plots/grouped/{label_prefix}/{group_name.replace(' ', '_')}_training_plot.png")
        plt.show()


def run_selected_models(
    models_to_run,
    X_train_tfidf,
    X_test_tfidf,
    X_train_bert,
    X_test_bert,
    train_texts,
    test_texts,
    train_labels,
    test_labels,
    label_names,
    label_prefix=""
):

    print(f"Running training using the following models: {models_to_run}")

    logreg_tfidf_results = []
    svm_tfidf_results = []
    logreg_bert_results = []
    svm_bert_results = []
    mlp_results = []
    cnn_results = []
    lstm_results = []
    bert_results = []

    history_dict = {}
    final_results = []

    num_epochs = 100

    def store_results(label, history, f1_train_micro, f1_train_macro, f1_test_micro, f1_test_macro,
                      best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy,
                      best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy,
                      results_array=None):
        
        label = f"{label_prefix.upper()} | {label}"

        entry = (
            label, f1_train_micro, f1_train_macro, f1_test_micro, f1_test_macro,
            best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy,
            best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy
        )
        if results_array is not None:
            results_array.append(entry)
        final_results.append(entry)
        history_dict[label] = history

    if "logreg_tfidf" in models_to_run:
        history, *metrics = run_logistic_regression_tfidf(X_train_tfidf, X_test_tfidf, train_labels, test_labels)
        store_results("LogReg TFIDF", history, *metrics, results_array=logreg_tfidf_results)

    if "svm_tfidf" in models_to_run:
        results = run_svm_tfidf(X_train_tfidf, X_test_tfidf, train_labels, test_labels)
        for label, (history, *metrics) in results.items():
            store_results(label, history, *metrics, results_array=svm_tfidf_results)

    if "logreg_bert" in models_to_run:
        history, *metrics = run_logistic_regression_bert(X_train_bert, X_test_bert, train_labels, test_labels)
        store_results("LogReg BERT", history, *metrics, results_array=logreg_bert_results)

    if "svm_bert" in models_to_run:
        results = run_svm_bert(X_train_bert, X_test_bert, train_labels, test_labels)
        for label, (history, *metrics) in results.items():
            store_results(label, history, *metrics, results_array=svm_bert_results)

    if "mlp_bert" in models_to_run:
        for dims, label in [([512, 256], "MLP 2-layer"), ([768, 512, 256], "MLP 3-layer")]:
            history, *metrics = run_mlp_bert(X_train_bert, X_test_bert, train_labels, test_labels, layer_dims=dims, label=label, num_epochs=num_epochs, label_names=label_names)
            store_results(label, history, *metrics, results_array=mlp_results)

    if "cnn_bert" in models_to_run:
        cnn_configs_list = [
            # ([(64, 3), (64, 5)], "cnn_small"),
            # ([(128, 3), (128, 4), (128, 5)], "cnn_medium"),
            # ([(256, 2), (256, 3), (256, 4)], "cnn_large"),
            # ([(64, 3), (64, 5)], "cnn_small_dropout_0.5", 0.5),
            # ([(128, 3), (128, 4), (128, 5)], "cnn_medium"),
            # ([(128, 3), (128, 4), (128, 5)], "cnn_medium_dropout_0.5", 0.5),
            # ([(256, 2), (256, 3), (256, 4)], "cnn_large"),
            # ([(256, 2), (256, 3), (256, 4)], "cnn_large_dropout_0.5", 0.5),
            # ([(256, 2), (256, 3), (256, 4)], "cnn_large_smaller_learning_rate", 0.3, 1e-5),
            # ([(256, 2), (256, 3), (256, 4)], "cnn_large_larger_learning_rate", 0.3, 1e-2),
            # ([(256, 2), (256, 3), (256, 4)], "cnn_large_weighted_bce", 0.3, 1e-4, 'weighted_bce'),
            # ([(256, 2), (256, 3), (256, 4)], "cnn_large_weighted_bce_dropout_0.5", 0.5, 1e-4, 'weighted_bce'),
            # ([(256, 2), (256, 3), (256, 4)], "cnn_large_weighted_bce_dropout_0.4", 0.4, 1e-4, 'weighted_bce'),
            # ([(256, 2), (256, 3), (256, 4)], "cnn_large_weighted_bce_dropout_0.6", 0.6, 1e-4, 'weighted_bce'),

            # ([(256, 2), (256, 3), (256, 4)], "cnn_large_weighted_bce_learning_rate_e-5", 0.3, 1e-5, 'weighted_bce'),
            ([(256, 2), (256, 3), (256, 4)], "cnn_large_weighted_bce_dropout_0.5_learning_rate_e-5", 0.5, 1e-5, 'weighted_bce'),
            ([(256, 2), (256, 3), (256, 4)], "cnn_large_weighted_bce_dropout_0.4_learning_rate_e-5", 0.4, 1e-5, 'weighted_bce'),
            # ([(256, 2), (256, 3), (256, 4)], "cnn_large_weighted_bce_dropout_0.6_learning_rate_e-5", 0.6, 1e-5, 'weighted_bce'),

            # ([(256, 2), (256, 3), (256, 4)], "cnn_large_weighted_bce_learning_rate_e-6", 0.3, 1e-6, 'weighted_bce'),
            ([(256, 2), (256, 3), (256, 4)], "cnn_large_weighted_bce_dropout_0.5_learning_rate_e-6", 0.5, 1e-6, 'weighted_bce'),
            ([(256, 2), (256, 3), (256, 4)], "cnn_large_weighted_bce_dropout_0.4_learning_rate_e-6", 0.4, 1e-6, 'weighted_bce'),
            # ([(256, 2), (256, 3), (256, 4)], "cnn_large_weighted_bce_dropout_0.6_learning_rate_e-6", 0.6, 1e-6, 'weighted_bce'),

            # ([(256, 2), (256, 3), (256, 4)], "cnn_large_focal", 0.3, 1e-4, 'focal'),
            # ([(128, 3), (128, 4), (128, 5)], "cnn_medium_weighted_bce", 0.3, 1e-4, 'weighted_bce'),
            # ([(128, 3), (128, 4), (128, 5)], "cnn_medium_weighted_bce_dropout_0.5", 0.5, 1e-4, 'weighted_bce'),
            # ([(128, 3), (128, 4), (128, 5)], "cnn_medium_weighted_bce_dropout_0.4", 0.4, 1e-4, 'weighted_bce'),
            # ([(128, 3), (128, 4), (128, 5)], "cnn_medium_weighted_bce_dropout_0.6", 0.6, 1e-4, 'weighted_bce'),
            # ([(128, 3), (128, 4), (128, 5)], "cnn_medium_focal", 0.3, 1e-4, 'focal'),
            ]
                
        for config in cnn_configs_list:
            if len(config) == 2:
                conv_configs, label = config
                dropout = 0.3
                lr = 1e-4
                loss_type = 'bce'
            elif len(config) == 3:
                conv_configs, label, dropout = config
                lr = 1e-4
                loss_type = 'bce'
            elif len(config) == 4:
                conv_configs, label, dropout, lr = config
                loss_type = 'bce'
            elif len(config) == 5:
                conv_configs, label, dropout, lr, loss_type = config
            else:
                raise ValueError("cnn_configs_list format is incorrect")
            
            history, *metrics = run_cnn_bert(
                X_train_bert, X_test_bert, train_labels, test_labels,
                conv_configs, label, num_epochs,
                dropout=dropout, lr=lr, loss_type=loss_type, label_names=label_names
            )
            store_results(label, history, *metrics, results_array=cnn_results)

    if "lstm_bert" in models_to_run:
        lstm_configs_list = [
            (256, 2, 0.3, 1e-4, "bilstm_default"),
            # (128, 2, 0.3, 1e-4, "bilstm_small_hidden"),
            (512, 2, 0.3, 1e-4, "bilstm_large_hidden"),
            (256, 3, 0.3, 1e-4, "bilstm_more_layers"),
            (256, 2, 0.5, 1e-4, "bilstm_more_dropout"),
            (256, 2, 0.3, 1e-5, "bilstm_lr_1e5"),
            (256, 2, 0.3, 1e-6, "bilstm_lr_1e6"),
        ]
        for hidden_dim, num_layers, dropout, lr, label in lstm_configs_list:
            print(f"\nRunning BiLSTM: {label}")
            history, *metrics = run_lstm_bert(X_train_bert, X_test_bert, train_labels, test_labels,
                                              hidden_dim, num_layers, dropout, lr, label, num_epochs, label_names=label_names)
            store_results(f"BiLSTM {label}", history, *metrics, results_array=lstm_results)

    if "bert_finetune" in models_to_run:
        bert_configs_list = [
            (2e-5, 0.3, "finetune_default"),  # (learning_rate, dropout, label)
            (1e-5, 0.3, "finetune_lr_1e5"),
            (1e-6, 0.3, "finetune_lr_1e6"),
            (1e-6, 0.4, "finetune_dropout_0.4"),
            # (1e-6, 0.5, "finetune_dropout_0.5"),
        ]   
         
        for lr, dropout, label in bert_configs_list:
            history, *metrics = run_finetuned_bert_model(train_texts, test_texts, train_labels, test_labels,
                                                         label, num_epochs, dropout, lr, label_names=label_names)
            store_results(f"BERT {label}", history, *metrics, results_array=bert_results)

    print("\n================ FINAL CONSOLIDATED MODEL COMPARISON (FULL) ====================\n")
    print(f"{'Model':<35} | {'Train Micro':>11} | {'Train Macro':>11} | {'Test Micro':>10} | {'Test Macro':>10} | "
        f"{'Best Ep (Acc)':>13} | {'Best F1 (Acc)':>13} | {'Best Acc (Acc)':>14} | "
        f"{'Best Ep (F1)':>13} | {'Best F1 (F1)':>13} | {'Best Acc (F1)':>14}")
    print("-" * 150)

    for entry in final_results:
        label, f1_train_micro, f1_train_macro, f1_test_micro, f1_test_macro, \
        best_epoch_acc, best_f1_acc, best_acc_acc, \
        best_epoch_f1, best_f1_f1, best_acc_f1 = entry

        print(f"{label:<35} | "
            f"{f1_train_micro:11.4f} | {f1_train_macro:11.4f} | {f1_test_micro:10.4f} | {f1_test_macro:10.4f} | "
            f"{best_epoch_acc:13} | {best_f1_acc:13.4f} | {best_acc_acc:14.4f} | "
            f"{best_epoch_f1:13} | {best_f1_f1:13.4f} | {best_acc_f1:14.4f}")

    
    # Save consolidated results to CSV
    import pandas as pd
    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame(final_results, columns=[
        "Model", "Train F1 Micro", "Train F1 Macro", "Test F1 Micro", "Test F1 Macro",
        "Best Epoch (Acc)", "Best F1 (Acc)", "Best Accuracy (Acc)",
        "Best Epoch (F1)", "Best F1 (F1)", "Best Accuracy (F1)"
    ])

    save_path = f"results/final_model_results_{label_prefix}.csv"
    results_df.to_csv(save_path, index=False)
    print(f"\nFinal consolidated results saved to: {save_path}")

    plot_all_model_histories(history_dict, label_prefix)
    plot_model_groups(history_dict, label_prefix)

    
    

###########################################################################################
# Run Models
###########################################################################################

# models_to_run = [
#     "logreg_tfidf",
#     "svm_tfidf",
#     "logreg_bert",
#     "svm_bert",
#     "mlp_bert",
#     "cnn_bert",
#     "lstm_bert",
#     "bert_finetune"
# ]

# run_selected_models(models_to_run, X_train_tfidf=X_train_tfidf,
#     X_test_tfidf=X_test_tfidf,
#     X_train_bert=X_train_bert,
#     X_test_bert=X_test_bert,
#     train_texts=train_texts,
#     test_texts=test_texts,
#     train_labels=train_labels,
#     test_labels=test_labels,
#     label_names= emotion_columns,
#     label_prefix="emotions")

