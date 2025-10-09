import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

from preprocessing.go_emotions_preprocessing import GoEmotionsPreprocessing
from utils.helper_methods import HelperMethods
from utils.logger import Logger


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
    helper_methods: HelperMethods = HelperMethods()
    go_emotions_preprocessing: GoEmotionsPreprocessing = GoEmotionsPreprocessing()

    def __init__(self, num_epochs: int = 100):
        self.logger: Logger = Logger(class_name=self.__class__.__name__)
        self.num_epochs = num_epochs

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

            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")

            # if test_f1 > best_f1:
            #     best_f1 = test_f1
            #     torch.save(model.state_dict(), save_path)
            #     self.logger.info(f"New best LSTM model saved for test F1: {test_f1:.4f}")

            best_epoch_test_accuracy = np.argmax(history["test_accuracy"])
            best_accuracy_f1_micro_test_accuracy = history["test_f1_micro"][best_epoch_test_accuracy]
            best_accuracy_test_accuracy = history["test_accuracy"][best_epoch_test_accuracy]

            best_epoch_f1_micro = np.argmax(history["test_f1_micro"])
            best_f1_micro_test_accuracy = history["test_f1_micro"][best_epoch_f1_micro]
            best_f1_test_accuracy = history["test_accuracy"][best_epoch_f1_micro]

        return history, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy

    def _run_lstm_bert(self, x_train_bert, x_test_bert, train_labels, test_labels, hidden_dimensions, num_layers,
                       dropout, learning_rate,
                       label,
                       num_epochs=200, label_names=""):
        self.logger.info(f"------------------ BiLSTM + BERT: {label} ------------------")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader, test_loader = self.helper_methods.prepare_mlp_dataloader(x_train_bert, x_test_bert, train_labels,
                                                                               test_labels)

        num_classes = train_labels.shape[1]
        model = BiLSTMClassifier(
            input_dim=768,
            hidden_dim=hidden_dimensions,
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
        #     self.logger.info(f"\nLoaded best BiLSTM model for {label}.")

        history, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy = self._train_lstm_model(
            model, train_loader, test_loader, device, num_epochs, save_path, lr=learning_rate, weights=weights)
        f1_micro_train, f1_macro_train, f1_micro_test, f1_macro_test = self.helper_methods.evaluate_cnn_model(model,
                                                                                                              train_loader,
                                                                                                              test_loader,
                                                                                                              device,
                                                                                                              label=label,
                                                                                                              label_names=label_names)

        return history, f1_micro_train, f1_macro_train, f1_micro_test, f1_macro_test, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy

    def execute_lstm_bert(self):

        lstm_results = []
        train_texts, test_texts, train_labels, test_labels = self.go_emotions_preprocessing.get_training_test_data_split()

        x_train_bert = self.helper_methods.get_bert_embeddings(train_texts, batch_size=32)
        x_test_bert = self.helper_methods.get_bert_embeddings(test_texts, batch_size=32)
        label_names = self.helper_methods.get_go_emotions_column_dataframe()

        lstm_configs_list = [
            (256, 2, 0.3, 1e-4, "bilstm_default"),
            # (128, 2, 0.3, 1e-4, "bilstm_small_hidden"),
            (512, 2, 0.3, 1e-4, "bilstm_large_hidden"),
            (256, 3, 0.3, 1e-4, "bilstm_more_layers"),
            (256, 2, 0.5, 1e-4, "bilstm_more_dropout"),
            (256, 2, 0.3, 1e-5, "bilstm_lr_1e5"),
            (256, 2, 0.3, 1e-6, "bilstm_lr_1e6"),
        ]
        for hidden_dimensions, num_layers, dropout, learning_rate, label in lstm_configs_list:
            self.logger.info(f"\nRunning BiLSTM: {label}")
            history, *metrics = self._run_lstm_bert(x_train_bert=x_train_bert, x_test_bert=x_test_bert,
                                                    train_labels=train_labels, test_labels=test_labels,
                                                    hidden_dimensions=hidden_dimensions, num_layers=num_layers,
                                                    dropout=dropout, learning_rate=learning_rate, label=label,
                                                    num_epochs=self.num_epochs,
                                                    label_names=label_names)
            self.helper_methods.store_results(f"BiLSTM {label}", history, *metrics, results_array=lstm_results)
