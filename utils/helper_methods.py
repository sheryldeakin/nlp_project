import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification

from nlp_project.utils.logger import Logger


class HelperMethods:

    def __init__(self):
        self.logger: Logger = Logger(class_name=self.__class__.__name__)

        self.bert_model = self.get_bert_model()
        self.auto_tokenizer = self.get_auto_tokenizer()
        self.emotion_column = self.get_go_emotions_column_dataframe()
        self.num_labels = len(self.emotion_column)

    def get_auto_tokenizer(self):
        return AutoTokenizer.from_pretrained("sdeakin/fine_tuned_bert_emotions")

    def get_bert_model(self):
        return BertForSequenceClassification.from_pretrained(
            "sdeakin/fine_tuned_bert_emotions",
            num_labels=self.num_labels,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True
        )

    def get_go_emotions_column_dataframe(self):

        go_emotions_dataframe: pd.DataFrame = pd.read_csv("resources/csv_files/go_emotions_dataset.csv")
        go_emotions_dataframe: pd.DataFrame = go_emotions_dataframe.drop(columns=["id", "example_very_unclear"])
        emotion_column = go_emotions_dataframe.columns[1:]

        return emotion_column

    def get_bert_embeddings(self, text_list, batch_size=128):
        embeddings = []

        total_batches = len(text_list) // batch_size + int(len(text_list) % batch_size != 0)

        self.logger.info(f" Total texts to process: {len(text_list)}")
        self.logger.info(f" Total batches expected: {total_batches}")

        with torch.no_grad():
            for i in tqdm(range(0, len(text_list), batch_size), desc="Processing BERT Embeddings"):
                if i + batch_size > len(text_list):  # Ensure last batch is fully processed
                    batch_texts = text_list[i:]
                else:
                    batch_texts = text_list[i: i + batch_size]

                # Tokenize batch
                # tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
                tokens = self.tokenizer(batch_texts, padding="max_length", truncation=True, return_tensors="pt",
                                        max_length=64)

                # Move tokens to GPU if available
                # tokens = {key: value.to(device) for key, value in tokens.items()}

                # Extract embeddings from BERT (use .bert to avoid classification head)
                outputs = self.bert_model.bert(**tokens)

                # Mean pooling: Average all token embeddings
                # batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                batch_embeddings = outputs.last_hidden_state.cpu().numpy()  # shape: [batch, seq_len, 768]
                embeddings.append(batch_embeddings)

        embeddings = np.vstack(embeddings)  # Stack all batches

        # **Check final shape**
        self.logger.info(f" Expected embeddings: {len(text_list)}, Extracted embeddings: {embeddings.shape[0]}")

        return embeddings
