import numpy as np
import pandas as pd
from preprocessing import text_clean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from utils.helper_methods import HelperMethods


class GoEmotionsPreprocessing:
    helper_methods: HelperMethods = HelperMethods()

    def __init__(self):
        self.file_path_str: str = "resources/csv_files/go_emotions_dataset.csv"

    def _get_dataframe(self) -> pd.DataFrame:
        return pd.read_csv(self.file_path_str)

    def _get_sample_dataframe(self) -> pd.DataFrame:
        dataframe: pd.DataFrame = self._get_dataframe()

        cleaned_dataframe: pd.DataFrame = dataframe.drop(columns=["id", "example_very_unclear"])

        # Get emotion columns (multi-label)
        emotion_columns = cleaned_dataframe.columns[1:]

        # Convert labels to lists of 0s and 1s (multi-label format)
        cleaned_dataframe[emotion_columns] = cleaned_dataframe[emotion_columns].astype(int)
        cleaned_dataframe["labels"] = cleaned_dataframe[emotion_columns].values.tolist()  # Store multi-label format

        # Keep only text and labels (REMOVE "label" since it does NOT exist)
        cleaned_dataframe = cleaned_dataframe[["text", "labels"]]

        # For initial testing, take a smaller sample
        df_sample: pd.DataFrame = cleaned_dataframe.sample(n=10000, random_state=42)

        # Apply text cleaning to dataset
        df_sample["cleaned_text"] = df_sample["text"].apply(text_clean.text_preprocessing_pipeline)

        return df_sample

    def get_training_test_data_split(self):
        df_sample: pd.DataFrame = self._get_sample_dataframe()

        train_texts, test_texts, train_labels, test_labels = train_test_split(
            df_sample["cleaned_text"], df_sample["labels"], test_size=0.2, random_state=42
        )

        return train_texts, test_texts, train_labels, test_labels

    def get_bert_embeddings_train_test_data(self):
        train_texts, test_texts, train_labels, test_labels = self.get_training_test_data_split()

        train_texts_list: list = train_texts.tolist()
        test_texts_list: list = test_texts.tolist()
        x_train_bert = self.helper_methods.get_bert_embeddings(train_texts_list, batch_size=32)
        x_test_bert = self.helper_methods.get_bert_embeddings(test_texts_list, batch_size=32)

        train_labels_array: np.ndarray = np.array(train_labels.tolist(), dtype=np.float32)
        test_labels_array: np.ndarray = np.array(test_labels.tolist(), dtype=np.float32)

        return x_train_bert, x_test_bert, train_labels_array, test_labels_array

    def get_tfidf_vectorized_train_test_data(self):
        train_texts, test_texts, train_labels, test_labels = self.get_training_test_data_split()

        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")

        train_texts_list: list = train_texts.tolist()
        test_texts_list: list = test_texts.tolist()
        train_labels_array: np.ndarray = np.array(train_labels.tolist(), dtype=np.float32)
        test_labels_array: np.ndarray = np.array(test_labels.tolist(), dtype=np.float32)

        x_train_tfidf = vectorizer.fit_transform(train_texts_list)
        x_test_tfidf = vectorizer.transform(test_texts_list)

        return x_train_tfidf, x_test_tfidf, train_labels_array, test_labels_array
