import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from nlp_project.utils.logger import Logger


class SVMTfidf:

    def __init__(self):
        self.logger: Logger = Logger(class_name=self.__class__.__name__)

    def run_svm_tfidf(self, X_train_tfidf, X_test_tfidf, train_labels, test_labels):
        self.logger.info("------------------ SVM + TFIDF ------------------")
        kernels = ["linear", "poly", "rbf", "sigmoid"]
        results = {}

        for kernel in kernels:
            self.logger.info(f"\nTraining SVM with {kernel} kernel...")
            clf = OneVsRestClassifier(SVC(kernel=kernel, degree=3, gamma="scale", probability=True))
            clf.fit(X_train_tfidf, train_labels)
            y_pred = clf.predict(X_test_tfidf)

            f1_micro = f1_score(test_labels, y_pred, average="micro")
            f1_macro = f1_score(test_labels, y_pred, average="macro")
            accuracy = accuracy_score(test_labels, y_pred)

            label = f"SVM TFIDF ({kernel})"
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

            self.logger.info(f"{label} | Acc: {accuracy:.4f} | F1 Micro: {f1_micro:.4f} | F1 Macro: {f1_macro:.4f}")
            results[label] = (history, f1_micro, f1_macro, f1_micro, f1_macro, best_epoch_test_accuracy,
                              best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro,
                              best_f1_micro_test_accuracy, best_f1_test_accuracy)

        return results  # dict of kernel: (history, ...)
