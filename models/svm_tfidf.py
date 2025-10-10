import numpy as np
from preprocessing.go_emotions_preprocessing import GoEmotionsPreprocessing
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from utils.helper_methods import HelperMethods
from utils.logger import Logger


class SVMTfidf:
    helper_methods: HelperMethods = HelperMethods()
    go_emotions_preprocessing: GoEmotionsPreprocessing = GoEmotionsPreprocessing()

    def __init__(self):
        self.logger: Logger = Logger(class_name=self.__class__.__name__)

    def run_svm_tfidf(self, X_train_tfidf, X_test_tfidf, train_labels, test_labels) -> dict:
        self.logger.info("=" * 100)
        self.logger.info("-" * 40 + "Training SVM + TFIDF" + "-" * 40)
        self.logger.info("=" * 100)

        kernels = ["linear", "poly", "rbf", "sigmoid"]
        results = {}

        for kernel in kernels:
            kernel_str: str = kernel.capitalize()
            self.logger.info(f"{'-' * 30} Training SVM with {kernel_str} Kernel {'-' * 30}")

            clf = OneVsRestClassifier(SVC(kernel=kernel, degree=3, gamma="scale", probability=True))
            clf.fit(X_train_tfidf, train_labels)
            y_pred = clf.predict(X_test_tfidf)

            f1_micro = f1_score(test_labels, y_pred, average="micro")
            f1_macro = f1_score(test_labels, y_pred, average="macro")
            accuracy = accuracy_score(test_labels, y_pred)

            label = f"SVM TFIDF ({kernel_str})"
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

            self.logger.info("=" * 100)
            self.logger.info(f"Label: {label}")
            self.logger.info(f"Accuracy: {accuracy:.4f}")
            self.logger.info(f"F1 Micro: {f1_micro:.4f}")
            self.logger.info(f"F1 Macro: {f1_macro:.4f}")
            self.logger.info("=" * 100)

            results[label] = (history, f1_micro, f1_macro, f1_micro, f1_macro, best_epoch_test_accuracy,
                              best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro,
                              best_f1_micro_test_accuracy, best_f1_test_accuracy)

        return results

    def execute_smv_tfidf_model(self) -> None:

        x_train_tfidf, x_test_tfidf, train_labels_array, test_labels_array = self.go_emotions_preprocessing.get_tfidf_vectorized_train_test_data()

        svm_tfidf_results: list = []

        results = self.run_svm_tfidf(x_train_tfidf, x_test_tfidf, train_labels_array, test_labels_array)
        for label, (history, *metrics) in results.items():
            self.helper_methods.store_results(label, history, *metrics, results_array=svm_tfidf_results)
