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


def train_cnn_model(model, train_loader, test_loader, device, num_epochs, save_path, lr=1e-4, loss_type="bce",
                    weights=0):
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

        print(
            f"Epoch {epoch + 1}/{num_epochs} - Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")

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


def run_cnn_bert(X_train_bert, X_test_bert, train_labels, test_labels, conv_configs, label, num_epochs, dropout=0.3,
                 lr=1e-4, loss_type='bce', label_names=""):
    print(f"------------------ CNN + BERT: {label} ------------------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = prepare_mlp_dataloader(X_train_bert, X_test_bert, train_labels, test_labels)

    num_classes = train_labels.shape[1]
    model = CNNClassifier(input_dim=768, num_classes=num_classes, conv_configs=conv_configs, dropout=dropout)
    save_path = f"resources/best_cnn_model_{label}.pt"

    # Compute class frequencies
    label_counts = np.sum(train_labels, axis=0)
    total = train_labels.shape[0]

    # Avoid divide-by-zero
    weights = total / (label_counts + 1e-6)
    weights = torch.tensor(weights, dtype=torch.float32)

    # if os.path.exists(save_path):
    #     model.load_state_dict(torch.load(save_path))
    #     print(f"\nLoaded best CNN model for {label}.")

    history, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy = train_cnn_model(
        model, train_loader, test_loader, device, num_epochs, save_path, lr=lr, loss_type=loss_type, weights=weights)
    f1_micro_train, f1_macro_train, f1_micro_test, f1_macro_test = evaluate_cnn_model(model, train_loader, test_loader,
                                                                                      device, label=label,
                                                                                      label_names=label_names)

    return history, f1_micro_train, f1_macro_train, f1_micro_test, f1_macro_test, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy
