from typing import Any

from models.logreg_bert import LogisticalRegressionBert
from models.lstm_bert_model import LSTMBert
from models.svm_tfidf import SVMTfidf
from utils.logger import Logger


class Controller:
    lstm_bert: LSTMBert = LSTMBert()
    svm_tfidf: SVMTfidf = SVMTfidf()
    logreg_bert: LogisticalRegressionBert = LogisticalRegressionBert()

    def __init__(self):
        self.logger: Logger = Logger(class_name=self.__class__.__name__)

    def execute_model(self, model_name_str: str) -> dict[str, Any]:

        model_dict: dict[str, Any] = {
            "mlp_bert": 0,
            "cnn_bert": 0,
            "svm_bert": 0,
            "lstm_bert": self.lstm_bert.execute_lstm_bert,
            "svm_tfidf": self.svm_tfidf.execute_smv_tfidf_model,
            "logreg_bert": self.logreg_bert.execute_logreg_bert_model,
            "logreg_tfidf": 0,
            "bert_finetune": 0,
            "all_models": 0

        }

        if model_name_str not in model_dict:
            self.logger.error(f"Model argument name passed {model_name_str} is not a supported model")
            raise Exception(f"Model argument name passed {model_name_str} is not a supported model")

        else:
            self.logger.info(f"Executing Model: {model_name_str.capitalize()}")
            model_dict[model_name_str]()

        return model_dict
