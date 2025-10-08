import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

from nlp_project.utils.helper_methods import HelperMethods
from nlp_project.utils.logger import Logger


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


class LSTMBert:

    def __init__(self):
        self.logger: Logger = Logger(class_name=self.__class__.__name__)
        self.helper_methods:HelperMethods = HelperMethods()

    def _train_lstm_model(self, model, train_loader, test_loader, device, num_epochs, save_path, lr=1e-8, weights=1):
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

            print(
                f"Epoch {epoch + 1}/{num_epochs} - Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")

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

    def _evaluate_cnn_model(self, model, train_loader, test_loader, device, label="cnn_model", label_names=""):
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
            print(f"\nPer-class metrics saved to {save_path}")

            # Print final sorted values
            print(f"\n{split_name} Set — Top Emotions by F1:")
            print(df.sort_values(by="F1", ascending=False).to_string(index=False))

            return f1_micro, f1_macro

        f1_micro_train, f1_macro_train = _evaluate(train_loader, "Train", label_names)
        f1_micro_test, f1_macro_test = _evaluate(test_loader, "Test", label_names)

        return f1_micro_train, f1_macro_train, f1_micro_test, f1_macro_test

    def _prepare_mlp_dataloader(self, X_train_bert, X_test_bert, train_labels, test_labels, batch_size=64):
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

    def _run_lstm_bert(self, X_train_bert, X_test_bert, train_labels, test_labels, hidden_dim, num_layers, dropout, lr,
                       label,
                       num_epochs=200, label_names=""):
        print(f"------------------ BiLSTM + BERT: {label} ------------------")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader, test_loader = self._prepare_mlp_dataloader(X_train_bert, X_test_bert, train_labels, test_labels)

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

        save_path = f"resources/best_lstm_model_{label}.pt"

        # if os.path.exists(save_path):
        #     model.load_state_dict(torch.load(save_path))
        #     print(f"\nLoaded best BiLSTM model for {label}.")

        history, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy = self._train_lstm_model(
            model, train_loader, test_loader, device, num_epochs, save_path, lr=lr, weights=weights)
        f1_micro_train, f1_macro_train, f1_micro_test, f1_macro_test = self._evaluate_cnn_model(model, train_loader,
                                                                                                test_loader,
                                                                                                device, label=label,
                                                                                                label_names=label_names)

        return history, f1_micro_train, f1_macro_train, f1_micro_test, f1_macro_test, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy

    def execute_lstm_bert(self):

        X_train_bert = self.helper_methods.get_bert_embeddings(train_texts, batch_size=32)

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
            history, *metrics = self._run_lstm_bert(X_train_bert, X_test_bert, train_labels, test_labels,
                                                    hidden_dim, num_layers, dropout, lr, label, num_epochs,
                                                    label_names=label_names)
            store_results(f"BiLSTM {label}", history, *metrics, results_array=lstm_results)
