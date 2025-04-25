import pandas as pd
import re
import os
import torch
from tqdm import tqdm

from transformers import pipeline

import json
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

CACHE_FILE = "sentiment_cache.json"

sentiment_cache = {}

# Load existing cache if available
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        sentiment_cache = json.load(f)

# Save cache after processing
def save_sentiment_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(sentiment_cache, f)

# Enable confidence scoring
sentiment_model = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english", 
    return_all_scores=True
)

def get_sentiment(text, threshold=0.98):
    scores = sentiment_model(text[:512])[0]  # list of dicts
    top = max(scores, key=lambda x: x["score"])
    return top["label"] if top["score"] >= threshold else None


def build_label_dict_with_sentiment(filepath, category_col, label_col, example_col):
    df = pd.read_csv(filepath)
    label_dict = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Building {filepath} labels"):
        category = row[category_col]
        label = row[label_col]
        example = str(row[example_col]) if pd.notna(row[example_col]) else ""

        if pd.notna(label):
            sentiment = get_sentiment(example)
            label_dict.setdefault(category, []).append((label.lower(), sentiment))  # Store label and sentiment

    return label_dict

# Load the Reddit dataset
reddit_df = pd.read_csv("data/reddit_mental_health_data.csv")
reddit_df["text"] = reddit_df["text"].astype(str).fillna("")

# Helper function to build label dictionary from CSV
def build_label_dict(filepath, category_col, label_col):
    df = pd.read_csv(filepath)
    print("Available columns:", df.columns.tolist())

    label_dict = {}
    for _, row in df.iterrows():
        category = row[category_col]
        label = row[label_col]
        if pd.notna(label):
            label = str(label).lower()
            label_dict.setdefault(category, []).append(label)
    return label_dict

def fuzzy_match(k):
    base = re.escape(k)
    return rf'\b{base}(ed|ing|er|s|es)?\b'

def label_text(text, label_dict, min_match_fraction=0.6):
    text = text.lower()
    matched = set()

    for category, keywords in label_dict.items():
        matched_keywords = sum(bool(re.search(fuzzy_match(k), text)) for k in keywords)
        # if matched_keywords / len(keywords) >= min_match_fraction:
        if matched_keywords > 0:
            matched.add(category)

    return list(matched) if matched else ["No Label Found"]

def label_text_with_sentiment(text, label_dict, min_match_fraction=0.6, sentiment_threshold=0.98):
    text = text.lower()
    text_sentiment = get_sentiment(text, threshold=sentiment_threshold)
    matched = set()

    for category, keyword_sentiments in label_dict.items():
        keyword_hits = sum(bool(re.search(fuzzy_match(k), text)) for k, _ in keyword_sentiments)
        sentiment_hits = sum(
            s == text_sentiment and text_sentiment is not None for _, s in keyword_sentiments
        )

        if keyword_hits / len(keyword_sentiments) >= min_match_fraction:
            matched.add(category)
        elif sentiment_hits / len(keyword_sentiments) >= 0.8:
            matched.add(category)

    return list(matched) if matched else ["No Label Found"]


def process_labels(
    reddit_df,
    sheet_path,
    category_col,
    label_col,
    output_name,
    use_sentiment=False,
    sentiment_threshold=0.98,
    min_match_fraction=0.6
):
    example_col = "Example"
    
    if use_sentiment:
        label_dict = build_label_dict_with_sentiment(sheet_path, category_col, label_col, example_col)
        label_func = lambda text: label_text_with_sentiment(
            text, label_dict, min_match_fraction=min_match_fraction, sentiment_threshold=sentiment_threshold
        )
    else:
        label_dict = build_label_dict(sheet_path, category_col, label_col)
        label_func = lambda text: label_text(text, label_dict, min_match_fraction=min_match_fraction)

    label_column = f"{output_name}_label_sentiment" if use_sentiment else f"{output_name}_label_keywords"

    reddit_df[label_column] = list(
        tqdm(reddit_df["text"].apply(label_func), total=len(reddit_df), desc=f"Labeling {output_name} ({'sentiment' if use_sentiment else 'keywords'})")
    )
    labeled_df = reddit_df[reddit_df[label_column].apply(lambda x: x != ["No Label Found"])].copy()
    labeled_df = labeled_df.explode(label_column)

    os.makedirs("labeled_outputs", exist_ok=True)
    suffix = "sentiment" if use_sentiment else "keywords"
    output_path = os.path.join("labeled_outputs", f"labeled_{output_name}_{suffix}.csv")
    labeled_df.to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path} ({len(labeled_df)} rows)")
    return labeled_df


def print_sample_labels(df, label_col, n=5):
    print(f"\n📌 Sample entries for {label_col}:\n" + "-"*50)
    for label in df[label_col].unique():
        if label != "No Label Found":
            print(f"\n🔹 Label: {label}")
            sample = df[df[label_col] == label]["text"].sample(n=min(n, df[df[label_col] == label].shape[0]), random_state=42)
            for i, txt in enumerate(sample):
                print(f"  {i+1}. {txt[:200]}{'...' if len(txt) > 200 else ''}")  # Truncate long examples


# Run the pipeline for each category
# Keyword-based labeling
triggers_keywords = process_labels(reddit_df, "data/Triggers - Sheet1.csv", "Trigger Categories", "Triggers", "triggers", use_sentiment=False)
themes_keywords = process_labels(reddit_df, "data/Themes - Sheet1.csv", "Theme Categories", "Themes", "themes", use_sentiment=False)
symptoms_keywords = process_labels(reddit_df, "data/Symptoms - Sheet1.csv", "Symptom Categories", "Symptoms", "symptoms", use_sentiment=False)

# Sentiment-augmented labeling
triggers_sentiment = process_labels(reddit_df, "data/Triggers - Sheet1.csv", "Trigger Categories", "Triggers", "triggers", use_sentiment=True)
themes_sentiment = process_labels(reddit_df, "data/Themes - Sheet1.csv", "Theme Categories", "Themes", "themes", use_sentiment=True)
symptoms_sentiment = process_labels(reddit_df, "data/Symptoms - Sheet1.csv", "Symptom Categories", "Symptoms", "symptoms", use_sentiment=True)


print_sample_labels(triggers_keywords, "triggers_label_keywords")
print_sample_labels(triggers_sentiment, "triggers_label_sentiment")

print_sample_labels(themes_keywords, "themes_label_keywords")
print_sample_labels(themes_sentiment, "themes_label_sentiment")

print_sample_labels(symptoms_keywords, "symptoms_label_keywords")
print_sample_labels(symptoms_sentiment, "symptoms_label_sentiment")


save_sentiment_cache()

reddit_df["triggers_label"].apply(lambda x: len(x) if isinstance(x, list) else 0).value_counts()

