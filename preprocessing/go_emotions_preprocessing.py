import pandas as pd
from preprocessing import text_clean
from sklearn.model_selection import train_test_split


class GoEmotionsPreprocessing:

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
