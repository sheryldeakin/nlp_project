def run_logistic_regression_bert(X_train_bert, X_test_bert, train_labels, test_labels):
    print("------------------ Logistic Regression + BERT ------------------")

    # === reshape to 2D via mean pooling ===
    X_train_flat = X_train_bert.mean(axis=1)  # shape: [num_samples, hidden_dim]
    X_test_flat = X_test_bert.mean(axis=1)

    train_labels_multiclass = np.argmax(train_labels, axis=1)
    test_labels_multiclass = np.argmax(test_labels, axis=1)


    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_flat, train_labels_multiclass)
    y_pred = clf.predict(X_test_flat)

    f1_micro = f1_score(test_labels_multiclass, y_pred, average="micro")
    f1_macro = f1_score(test_labels_multiclass, y_pred, average="macro")
    accuracy = accuracy_score(test_labels_multiclass, y_pred)

    history = {
        "train_loss": [0],
        "train_f1_micro": [f1_micro],
        "test_f1_micro": [f1_micro],
        "train_accuracy": [accuracy],
        "test_accuracy": [accuracy],
    }

    best_epoch_test_accuracy = np.argmax(history["test_accuracy"])
    best_accuracy_f1_micro_test_accuracy = history["test_f1_micro"][best_epoch_test_accuracy]
    best_accuracy_test_accuracy = history["test_accuracy"][best_epoch_test_accuracy]

    best_epoch_f1_micro = np.argmax(history["test_f1_micro"])
    best_f1_micro_test_accuracy = history["test_f1_micro"][best_epoch_f1_micro]
    best_f1_test_accuracy = history["test_accuracy"][best_epoch_f1_micro]   
 
    print(f"LogReg + BERT | Acc: {accuracy:.4f} | F1 Micro: {f1_micro:.4f} | F1 Macro: {f1_macro:.4f}")
    return history, f1_micro, f1_macro, f1_micro, f1_macro, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy