def run_logistic_regression_tfidf(X_train_tfidf, X_test_tfidf, train_labels, test_labels):
    print("------------------ Logistic Regression + TFIDF ------------------")

    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X_train_tfidf, train_labels)
    y_pred = clf.predict(X_test_tfidf)

    # Evaluate
    f1_micro = f1_score(test_labels, y_pred, average="micro")
    f1_macro = f1_score(test_labels, y_pred, average="macro")
    accuracy = accuracy_score(test_labels, y_pred)

    history = {
        "train_loss": [0],
        "train_f1_micro": [f1_micro],  # Placeholder for compatibility
        "test_f1_micro": [f1_micro],
        "train_accuracy": [accuracy],  # Placeholder
        "test_accuracy": [accuracy],
    }

    best_epoch_test_accuracy = np.argmax(history["test_accuracy"])
    best_accuracy_f1_micro_test_accuracy = history["test_f1_micro"][best_epoch_test_accuracy]
    best_accuracy_test_accuracy = history["test_accuracy"][best_epoch_test_accuracy]

    best_epoch_f1_micro = np.argmax(history["test_f1_micro"])
    best_f1_micro_test_accuracy = history["test_f1_micro"][best_epoch_f1_micro]
    best_f1_test_accuracy = history["test_accuracy"][best_epoch_f1_micro]   

    print(f"LogReg + TFIDF | Acc: {accuracy:.4f} | F1 Micro: {f1_micro:.4f} | F1 Macro: {f1_macro:.4f}")
    return history, f1_micro, f1_macro, f1_micro, f1_macro, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy