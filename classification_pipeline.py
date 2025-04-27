import time
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm

from emotions_multi_class_classifier import run_selected_models, get_bert_embeddings
from sklearn.feature_extraction.text import TfidfVectorizer


def load_and_prepare_dataset(path, label_col, prefix):
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str)
    
    # Normalize and prefix label
    df[label_col] = df[label_col].fillna("").astype(str)
    df = df[df[label_col] != ""]
    df[label_col] = df[label_col].apply(lambda x: [f"{prefix}_{x.strip()}"] if isinstance(x, str) else [])
    
    # Explode and regroup to get multilabels
    df = df.groupby("text")[label_col].sum().reset_index()
    
    # Binarize
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(df[label_col])
    label_names = mlb.classes_

    print(f"{prefix} dataset: {len(df)} examples, {len(label_names)} labels")

    return df["text"].tolist(), labels, label_names

def prepare_bert_embeddings(texts, batch_size=32):
    return get_bert_embeddings(texts, batch_size=batch_size)

def process_and_run_all_models():
    print(f"RUNNING THE PIPELINE")
    datasets = [
        # # TRIGGERS
        ("labeled_outputs/labeled_triggers_keywords.csv", "triggers_label_keywords", "triggers_keywords"),
        ("labeled_outputs/labeled_triggers_sentiment.csv", "triggers_label_sentiment", "triggers_sentiment"),

        # THEMES
        ("labeled_outputs/labeled_themes_keywords.csv", "themes_label_keywords", "themes_keywords"),
        ("labeled_outputs/labeled_themes_sentiment.csv", "themes_label_sentiment", "themes_sentiment"),

        # SYMPTOMS
        ("labeled_outputs/labeled_symptoms_keywords.csv", "symptoms_label_keywords", "symptoms_keywords"),
        ("labeled_outputs/labeled_symptoms_sentiment.csv", "symptoms_label_sentiment", "symptoms_sentiment"),
    ]

    progress = tqdm(datasets, desc="Running datasets", unit="dataset")

    for path, label_col, prefix in progress:
        start_time = time.time()
        
        texts, labels, label_names = load_and_prepare_dataset(path, label_col, prefix)

        # Split
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        # Get BERT embeddings
        print(f"🔄 Generating BERT embeddings for {prefix}...")
        X_train_bert = prepare_bert_embeddings(train_texts)
        X_test_bert = prepare_bert_embeddings(test_texts)

        # TFIDF (optional - skip if only using BERT)
        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        X_train_tfidf = vectorizer.fit_transform(train_texts)
        X_test_tfidf = vectorizer.transform(test_texts)

        # Set globals (you can modularize this if you want isolation)
        globals()["X_train_bert"] = X_train_bert
        globals()["X_test_bert"] = X_test_bert
        globals()["X_train_tfidf"] = X_train_tfidf
        globals()["X_test_tfidf"] = X_test_tfidf
        globals()["train_texts"] = train_texts
        globals()["test_texts"] = test_texts
        globals()["train_labels"] = np.array(train_labels, dtype=np.float32)
        globals()["test_labels"] = np.array(test_labels, dtype=np.float32)

        models_to_run = [
            "logreg_tfidf",
            "svm_tfidf",
            "logreg_bert",
            "svm_bert",
            "mlp_bert",
            "cnn_bert",
            "lstm_bert",
            "bert_finetune"
        ]

        print(f"Running models for {prefix.upper()}")
        run_selected_models(models_to_run, X_train_tfidf=X_train_tfidf,
            X_test_tfidf=X_test_tfidf,
            X_train_bert=X_train_bert,
            X_test_bert=X_test_bert,
            train_texts=train_texts,
            test_texts=test_texts,
            train_labels=train_labels,
            test_labels=test_labels,
            label_names= label_names,
            label_prefix=prefix)

        # Show time per dataset
        elapsed = time.time() - start_time
        print(f"✅ Finished {prefix} in {elapsed / 60:.1f} minutes")

if __name__ == "__main__":
    process_and_run_all_models()
