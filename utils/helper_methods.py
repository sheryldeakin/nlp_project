import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification

from nlp_project.utils.logger import Logger


class HelperMethods:

    def __init__(self):
        self.logger: Logger = Logger(class_name=self.__class__.__name__)

        self.bert_model = self.get_bert_model()
        self.auto_tokenizer = self.get_auto_tokenizer()
        self.emotion_column = self.get_go_emotions_column_dataframe()
        self.num_labels = len(self.emotion_column)

    def get_auto_tokenizer(self):
        return AutoTokenizer.from_pretrained("sdeakin/fine_tuned_bert_emotions")

    def get_bert_model(self):
        return BertForSequenceClassification.from_pretrained(
            "sdeakin/fine_tuned_bert_emotions",
            num_labels=self.num_labels,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True
        )

    def get_go_emotions_column_dataframe(self):

        go_emotions_dataframe: pd.DataFrame = pd.read_csv("resources/csv_files/go_emotions_dataset.csv")
        go_emotions_dataframe: pd.DataFrame = go_emotions_dataframe.drop(columns=["id", "example_very_unclear"])
        emotion_column = go_emotions_dataframe.columns[1:]

        return emotion_column

    def prepare_mlp_dataloader(self, X_train_bert, X_test_bert, train_labels, test_labels, batch_size=64):
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

    def store_results(self, label, history, f1_train_micro, f1_train_macro, f1_test_micro, f1_test_macro,
                      best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy,
                      best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy,
                      results_array=None):

        label_prefix = "emotions"

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

    def get_bert_embeddings(self, text_list, batch_size=128):
        embeddings = []

        total_batches = len(text_list) // batch_size + int(len(text_list) % batch_size != 0)

        self.logger.info(f"Total texts to process: {len(text_list)}")
        self.logger.info(f"Total batches expected: {total_batches}")

        with torch.no_grad():
            for i in tqdm(range(0, len(text_list), batch_size), desc="Processing BERT Embeddings"):
                if i + batch_size > len(text_list):  # Ensure last batch is fully processed
                    batch_texts = text_list[i:]
                else:
                    batch_texts = text_list[i: i + batch_size]

                # Tokenize batch
                # tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
                tokens = self.tokenizer(batch_texts, padding="max_length", truncation=True, return_tensors="pt",
                                        max_length=64)

                # Move tokens to GPU if available
                # tokens = {key: value.to(device) for key, value in tokens.items()}

                # Extract embeddings from BERT (use .bert to avoid classification head)
                outputs = self.bert_model.bert(**tokens)

                # Mean pooling: Average all token embeddings
                # batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                batch_embeddings = outputs.last_hidden_state.cpu().numpy()  # shape: [batch, seq_len, 768]
                embeddings.append(batch_embeddings)

        embeddings = np.vstack(embeddings)  # Stack all batches

        # **Check final shape**
        self.logger.info(f" Expected embeddings: {len(text_list)}, Extracted embeddings: {embeddings.shape[0]}")

        return embeddings

    def evaluate_cnn_model(self, model, train_loader, test_loader, device, label="cnn_model", label_names=""):
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

            self.logger.info(f"\n{split_name} Set Metrics for CNN + BERT:")
            self.logger.info(f"Overall Accuracy: {acc:.4f}")
            self.logger.info(f"F1 Score (Micro): {f1_micro:.4f}")
            self.logger.info(f"F1 Score (Macro): {f1_macro:.4f}")

            y_true = np.array(all_labels)
            y_pred = np.array(all_preds)

            # Per-class metrics
            precisions, recalls, f1s, _ = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )

            if isinstance(label_names, (list, np.ndarray)):
                emotion_labels = label_names.tolist() if isinstance(label_names, np.ndarray) else label_names
            else:
                raise ValueError(
                    f"Expected label_names to be a list or ndarray, but got: {type(label_names).__name__}, label_names: {label_names}")

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
            self.logger.info(f"\nPer-class metrics saved to {save_path}")

            # Print final sorted values
            self.logger.info(f"\n{split_name} Set — Top Emotions by F1:")
            self.logger.info(df.sort_values(by="F1", ascending=False).to_string(index=False))

            return f1_micro, f1_macro

        f1_micro_train, f1_macro_train = _evaluate(train_loader, "Train", label_names)
        f1_micro_test, f1_macro_test = _evaluate(test_loader, "Test", label_names)

        return f1_micro_train, f1_macro_train, f1_micro_test, f1_macro_test
