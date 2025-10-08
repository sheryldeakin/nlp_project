import numpy as np
import pandas as pd

from preprocessing import text_clean
from utils import constants

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

# Load dataset
df = pd.read_csv("resources/csv_files/go_emotions_dataset.csv")  # Update with the correct path

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


# Apply text cleaning to dataset
df_sample["cleaned_text"] = df_sample["text"].apply(text_clean.text_preprocessing_pipeline)

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



###########################################################################################
# SVM
###########################################################################################

from sklearn.multiclass import OneVsRestClassifier


###########################################################################################
# Using BERT to expand text instead of TF-IDF
###########################################################################################
from transformers import BertModel
from tqdm import tqdm
import numpy as np

from transformers import AutoTokenizer

# Load pre-trained BERT tokenizer & model - Baseline BERT model
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# bert_model = BertModel.from_pråetrained("bert-base-uncased")


################
################ NONLINEAR MODELS NEEDS THE BERT EMBEDDINGS
################

num_labels = len(emotion_columns)
print(f"num_labels: {num_labels}")


# ###########################################################################################
# # BERT + Logistic Regression
# ###########################################################################################

from sklearn.linear_model import LogisticRegression


###########################################################################################
# SVM + BERT
###########################################################################################

from sklearn.svm import SVC


###########################################################################################
# MLP + BERT (Multiclass)
###########################################################################################

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ###########################################################################################
# # CNN + BERT 
# ###########################################################################################
import torch.nn.functional as F

import torch
import torch.nn as nn


# ###########################################################################################
# # BiLSTM + BERT
# ###########################################################################################

# ###########################################################################################
# # Straight up BERT
# ###########################################################################################

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
            history, *metrics = run_mlp_bert(X_train_bert, X_test_bert, train_labels, test_labels, layer_dims=dims,
                                             label=label, num_epochs=num_epochs, label_names=label_names)
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
            ([(256, 2), (256, 3), (256, 4)], "cnn_large_weighted_bce_dropout_0.5_learning_rate_e-5", 0.5, 1e-5,
             'weighted_bce'),
            ([(256, 2), (256, 3), (256, 4)], "cnn_large_weighted_bce_dropout_0.4_learning_rate_e-5", 0.4, 1e-5,
             'weighted_bce'),
            # ([(256, 2), (256, 3), (256, 4)], "cnn_large_weighted_bce_dropout_0.6_learning_rate_e-5", 0.6, 1e-5, 'weighted_bce'),

            # ([(256, 2), (256, 3), (256, 4)], "cnn_large_weighted_bce_learning_rate_e-6", 0.3, 1e-6, 'weighted_bce'),
            ([(256, 2), (256, 3), (256, 4)], "cnn_large_weighted_bce_dropout_0.5_learning_rate_e-6", 0.5, 1e-6,
             'weighted_bce'),
            ([(256, 2), (256, 3), (256, 4)], "cnn_large_weighted_bce_dropout_0.4_learning_rate_e-6", 0.4, 1e-6,
             'weighted_bce'),
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
                                              hidden_dim, num_layers, dropout, lr, label, num_epochs,
                                              label_names=label_names)
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
