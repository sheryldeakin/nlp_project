def run_svm_bert(X_train_bert, X_test_bert, train_labels, test_labels):
    print("------------------ SVM + BERT ------------------")
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    results = {}

    X_train_flat = X_train_bert.mean(axis=1)
    X_test_flat = X_test_bert.mean(axis=1)


    for kernel in kernels:
        print(f"\nTraining SVM + BERT with {kernel} kernel...")
        clf = OneVsRestClassifier(SVC(kernel=kernel, degree=3, gamma="scale", probability=True))
        clf.fit(X_train_flat, train_labels)
        y_pred = clf.predict(X_test_flat)

        f1_micro = f1_score(test_labels, y_pred, average="micro")
        f1_macro = f1_score(test_labels, y_pred, average="macro")
        accuracy = accuracy_score(test_labels, y_pred)

        label = f"SVM BERT ({kernel})"
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

        print(f"{label} | Acc: {accuracy:.4f} | F1 Micro: {f1_micro:.4f} | F1 Macro: {f1_macro:.4f}")
        results[label] = (history, f1_micro, f1_macro, f1_micro, f1_macro, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy)

    return results