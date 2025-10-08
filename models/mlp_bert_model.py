import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset


class MLPClassifier(nn.Module):

    def __init__(self, input_dim, layer_dims, num_classes):
        super(MLPClassifier, self).__init__()

        layers = []
        current_dim = input_dim
        for dim in layer_dims:
            layers.append(nn.Linear(current_dim, dim))
            layers.append(nn.BatchNorm1d(dim))  # USE BATCH NORM 1 or 2D check both
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))  # Don't use drop out originally, unless I get good enough results
            current_dim = dim

        layers.append(nn.Linear(current_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MLPBert:

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

    def _evaluate_mlp_model(self, model, train_loader, test_loader, device, label_names):
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
                raise ValueError(
                    f"Expected label_names to be a list or ndarray, but got: {type(label_names).__name__}, label_names: {label_names}")

            print("\nPer-class Precision / Recall / F1:")
            for idx, label in enumerate(emotion_labels):  # or custom label list
                print(f"{label:20s} | P: {precisions[idx]:.2f} | R: {recalls[idx]:.2f} | F1: {f1s[idx]:.2f}")

            return mlp_f1_micro, mlp_f1_macro

    def run_mlp_bert(self, X_train_bert, X_test_bert, train_labels, test_labels, layer_dims, label, num_epochs,
                     label_names):
        print(f"------------------ MLP + BERT: {label} ------------------")

        X_train_flat = X_train_bert.mean(axis=1)  # [num_samples, hidden_dim]
        X_test_flat = X_test_bert.mean(axis=1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader, test_loader = self._prepare_mlp_dataloader(
            X_train_flat, X_test_flat, train_labels, test_labels
        )

        num_classes = train_labels.shape[1]
        model = MLPClassifier(input_dim=768, layer_dims=layer_dims, num_classes=num_classes)

        save_path = f"resources/best_mlp_model_{label}.pt"

        history, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy = train_mlp_model(
            model, train_loader, test_loader, device, num_epochs, save_path)
        mlp_f1_micro_train, mlp_f1_macro_train, mlp_f1_micro_test, mlp_f1_macro_test = self._evaluate_mlp_model(model,
                                                                                                                train_loader,
                                                                                                                test_loader,
                                                                                                                device,
                                                                                                                label_names=label_names)

        return history, mlp_f1_micro_train, mlp_f1_macro_train, mlp_f1_micro_test, mlp_f1_macro_test, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy
