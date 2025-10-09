import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import BertModel
from transformers import BertTokenizer, BertForSequenceClassification

from nlp_project.utils.logger import Logger


def tokenize_for_bert(texts, tokenizer, max_length=64):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return encodings["input_ids"], encodings["attention_mask"]


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


class BertFinetune:

    def __init__(self):
        self.logger: Logger = Logger(class_name=self.__class__.__name__)

    def train_bert_finetune_model(self, model, train_loader, test_loader, device, num_epochs, save_path, lr=2e-5,
                                  weights=1):
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weights.to(device))
        model.to(device)

        best_f1 = 0
        history = {"train_loss": [], "train_f1_micro": [], "test_f1_micro": [], "train_accuracy": [],
                   "test_accuracy": []}

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

            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")

            # if test_f1 > best_f1:
            #     best_f1 = test_f1
            #     torch.save(model.state_dict(), save_path)
            #     self.logger.info(f"New best BERT model saved for test F1: {test_f1:.4f}")

            best_epoch_test_accuracy = np.argmax(history["test_accuracy"])
            best_accuracy_f1_micro_test_accuracy = history["test_f1_micro"][best_epoch_test_accuracy]
            best_accuracy_test_accuracy = history["test_accuracy"][best_epoch_test_accuracy]

            best_epoch_f1_micro = np.argmax(history["test_f1_micro"])
            best_f1_micro_test_accuracy = history["test_f1_micro"][best_epoch_f1_micro]
            best_f1_test_accuracy = history["test_accuracy"][best_epoch_f1_micro]

        return history, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy

    def evaluate_bert_model(self, model, train_loader, test_loader, device, label="bert_finetune", label_names=""):
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

            self.logger.info(f"\n{split_name} Set Metrics for {label}:")
            self.logger.info(f" Accuracy: {acc:.4f}")
            self.logger.info(f" F1 Score (Micro): {f1_micro:.4f}")
            self.logger.info(f" F1 Score (Macro): {f1_macro:.4f}")

            y_true = np.array(all_labels)
            y_pred = np.array(all_preds)

            precisions, recalls, f1s, _ = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )

            if isinstance(label_names, (list, np.ndarray)):
                emotion_labels = label_names.tolist() if isinstance(label_names, np.ndarray) else label_names
            else:
                raise ValueError(
                    f"Expected label_names to be a list or ndarray, but got: {type(label_names).__name__}, label_names: {label_names}")

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
            self.logger.info(f"\nPer-class metrics saved to {save_path}")

            self.logger.info(f"\n🔍 {split_name} Set — Top Emotions by F1:")
            self.logger.info(df.sort_values(by="F1", ascending=False).to_string(index=False))

            return f1_micro, f1_macro, acc

        f1_micro_train, f1_macro_train, train_acc = _evaluate(train_loader, "Train", label_names)
        f1_micro_test, f1_macro_test, test_acc = _evaluate(test_loader, "Test", label_names)

        return f1_micro_train, f1_macro_train, f1_micro_test, f1_macro_test

    def run_finetuned_bert_model(self, train_texts, test_texts, train_labels, test_labels, label, num_epochs=200,
                                 dropout=0.3,
                                 lr=2e-5, label_names=""):
        self.logger.info(f"------------------ Fine-Tuned BERT: {label} ------------------")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BertForSequenceClassification.from_pretrained(
            "sdeakin/fine_tuned_bert_emotions",
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True
        )
        tokenizer = BertTokenizer.from_pretrained("sdeakin/fine_tuned_bert_emotions")

        train_loader, test_loader = prepare_bert_dataloaders(train_texts, test_texts, train_labels, test_labels,
                                                             tokenizer)

        # Replace classifier dynamically
        model.classifier = torch.nn.Linear(model.config.hidden_size, train_labels.shape[1])
        model.num_labels = train_labels.shape[1]

        # model = BERTClassifier("fine_tuned_bert_emotions", num_classes=train_labels.shape[1], dropout=dropout)
        save_path = f"resources/best_finetuned_bert_model_{label}.pt"

        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path))
            self.logger.info(f"\nLoaded best BERT model for {label}.")

        # Compute class frequencies
        label_counts = np.sum(train_labels, axis=0)
        total = train_labels.shape[0]

        # Avoid divide-by-zero
        weights = total / (label_counts + 1e-6)
        weights = torch.tensor(weights, dtype=torch.float32)

        history, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy = self.train_bert_finetune_model(
            model, train_loader, test_loader, device, num_epochs, save_path, lr=lr, weights=weights)

        f1_micro_train, f1_macro_train, f1_micro_test, f1_macro_test = self.evaluate_bert_model(
            model, train_loader, test_loader, device, label=label, label_names=label_names
        )

        return history, f1_micro_train, f1_macro_train, f1_micro_test, f1_macro_test, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy
