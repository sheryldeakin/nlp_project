import numpy as np
from preprocessing.go_emotions_preprocessing import GoEmotionsPreprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from utils.helper_methods import HelperMethods
from utils.logger import Logger


class LogisticalRegressionBert:
    helper_methods: HelperMethods = HelperMethods()
    go_emotions_preprocessing: GoEmotionsPreprocessing = GoEmotionsPreprocessing()

    def __init__(self):
        self.logger: Logger = Logger(class_name=self.__class__.__name__)

    def run_logistic_regression_bert(self, x_train_bert, x_test_bert, train_labels_array, test_labels_array):
        self.logger.info("------------------ Logistic Regression + BERT ------------------")

        # === reshape to 2D via mean pooling ===
        X_train_flat = x_train_bert.mean(axis=1)  # shape: [num_samples, hidden_dim]
        X_test_flat = x_test_bert.mean(axis=1)

        train_labels_multiclass = np.argmax(train_labels_array, axis=1)
        test_labels_multiclass = np.argmax(test_labels_array, axis=1)

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

        self.logger.info(f"LogReg + BERT | Acc: {accuracy:.4f} | F1 Micro: {f1_micro:.4f} | F1 Macro: {f1_macro:.4f}")
        return history, f1_micro, f1_macro, f1_micro, f1_macro, best_epoch_test_accuracy, best_accuracy_f1_micro_test_accuracy, best_accuracy_test_accuracy, best_epoch_f1_micro, best_f1_micro_test_accuracy, best_f1_test_accuracy

    def execute_logreg_bert_model(self) -> None:
        x_train_bert, x_test_bert, train_labels_array, test_labels_array = self.go_emotions_preprocessing.get_bert_embeddings_train_test_data()

        logreg_bert_results: list = []

        history, *metrics = self.run_logistic_regression_bert(x_train_bert=x_train_bert, x_test_bert=x_test_bert,
                                                              train_labels_array=train_labels_array,
                                                              test_labels_array=test_labels_array)
        self.helper_methods.store_results("LogReg BERT", history, *metrics, results_array=logreg_bert_results)
