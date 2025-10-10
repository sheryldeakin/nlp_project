import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from utils.helper_methods import HelperMethods
from utils.logger import Logger


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
    helper_methods: HelperMethods = HelperMethods()

    def __init__(self):
        self.logger: Logger = Logger(class_name=self.__class__.__name__)

    def _evaluate(self, model, loader, device, split_name, label_names):
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

        self.logger.info(f"\n{split_name} Set Metrics for MLP + BERT:")
        self.logger.info(f"\nOverall Accuracy: {mlp_accuracy:.4f}")
        self.logger.info(f"Overall F1 Score (Micro): {mlp_f1_micro:.4f}")
        self.logger.info(f"Overall F1 Score (Macro): {mlp_f1_macro:.4f}")

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

        self.logger.info("\nPer-class Precision / Recall / F1:")
        for idx, label in enumerate(emotion_labels):  # or custom label list
            self.logger.info(f"{label:20s} | P: {precisions[idx]:.2f} | R: {recalls[idx]:.2f} | F1: {f1s[idx]:.2f}")

        return mlp_f1_micro, mlp_f1_macro

    def _evaluate_mlp_model(self, model, train_loader, test_loader, device, label_names):

        # Evaluate both train and test
        mlp_f1_micro_train, mlp_f1_macro_train = self._evaluate(model=model, device=device, loader=train_loader,
                                                                split_name="Train", label_names=label_names)
        mlp_f1_micro_test, mlp_f1_macro_test = self._evaluate(model=model, device=device, loader=test_loader,
                                                              split_name="Test", label_names=label_names)

        return mlp_f1_micro_train, mlp_f1_macro_train, mlp_f1_micro_test, mlp_f1_macro_test

    def _train_mlp_model(self, model, train_loader, test_loader, device, num_epochs, save_path):
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
        self.logger.info("\n Training MLP Classifier on BERT Embeddings...")
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

            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - Loss: {total_loss:.4f} | Train Acc: {train_accuracy:.4f} | Train F1: {train_f1_micro:.4f} | Test Acc: {test_accuracy:.4f} | Test F1: {test_f1_micro:.4f}")

            # # Save best model
            # if test_f1_micro > best_f1:
            #     best_f1 = test_f1_micro
            #     torch.save(model.state_dict(), save_path)
            #     self.logger.info(f"New best model saved (Test F1 Micro: {test_f1_micro:.4f})")

            best_epoch_test_accuracy = np.argmax(history["test_accuracy"])
            best_accuracy_f1_micro_test_accuracy = history["test_f1_micro"][best_epoch_test_accuracy]
            best_accuracy_test_accuracy = history["test_accuracy"][best_epoch_test_accuracy]

            best_epoch_f1_micro = np.argmax(history["test_f1_micro"])
            best_f1_micro_test_accuracy = history["test_f1_micro"][best_epoch_f1_micro]
            best_f1_test_accuracy = history["test_accuracy"][best_epoch_f1_micro]

        return history, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy

    def run_mlp_bert(self, x_train_bert, x_test_bert, train_labels, test_labels, layer_dims, label, num_epochs,
                     label_names):
        self.logger.info(f"------------------ MLP + BERT: {label} ------------------")

        x_train_flat = x_train_bert.mean(axis=1)  # [num_samples, hidden_dim]
        x_test_flat = x_test_bert.mean(axis=1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader, test_loader = self.helper_methods.prepare_mlp_dataloader(
            x_train_flat, x_test_flat, train_labels, test_labels
        )

        num_classes = train_labels.shape[1]
        model = MLPClassifier(input_dim=768, layer_dims=layer_dims, num_classes=num_classes)

        save_path = f"resources/best_mlp_model_{label}.pt"

        history, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy = self._train_mlp_model(
            model, train_loader, test_loader, device, num_epochs, save_path)
        mlp_f1_micro_train, mlp_f1_macro_train, mlp_f1_micro_test, mlp_f1_macro_test = self._evaluate_mlp_model(model,
                                                                                                                train_loader,
                                                                                                                test_loader,
                                                                                                                device,
                                                                                                                label_names=label_names)

        return history, mlp_f1_micro_train, mlp_f1_macro_train, mlp_f1_micro_test, mlp_f1_macro_test, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy
