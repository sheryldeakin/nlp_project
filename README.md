# Reading Between the Lines: Multi-Label Classification of Mental Health Signals Using Weak Supervision and BERT Embeddings
# 
This project implements a full **multi-label classification pipeline** for detecting mental health **triggers**, **themes**, and **symptoms** from text data.

It supports training and evaluating a variety of models, including:
- TF-IDF + Logistic Regression
- TF-IDF + SVM (multiple kernels)
- BERT embeddings + Logistic Regression / SVM
- MLP trained on BERT embeddings
- CNN trained on BERT embeddings
- BiLSTM trained on BERT embeddings
- Fine-tuned BERT (end-to-end training)

The project compares all models using accuracy, F1 scores, and plots training performance over epochs.

---

## Project Structure

- `classification_pipeline.py`: Main script to run the full training and evaluation across all models and datasets.
- `emotions_multi_class_classifier.py`: Contains model architectures, training, and evaluation logic. Uncomment bottom to run for emotions (this realistically should be added to the pipeline but i ran out of time)
- `labeled_outputs/`: Folder expected to contain the labeled trigger, theme, and symptom datasets (explained below).
- `plots/` and `results/`: Output folders for saving model training graphs and final evaluation results.

---

## Important Note: Generating Labeled Outputs

The `labeled_outputs/` CSV files (e.g., `labeled_triggers_keywords.csv`, `labeled_themes_keywords.csv`, etc.) are **not included** by default due to size and github restrictions.

You must **generate these files** before running `classification_pipeline.py`.

These labeled outputs are created by running your own labeling script or model that assigns:
- Keyword-based trigger labels
- Sentiment-based trigger labels
- Keyword-based theme labels
- Sentiment-based theme labels
- Keyword-based symptom labels
- Sentiment-based symptom labels

Each generated CSV should contain:
| Column | Description |
|:-------|:------------|
| `text` | Raw input text |
| `<label_column>` | Label assigned (e.g., `"triggers_label_keywords"`) |

---

## 🛠️ Setup Instructions

1. **Clone the repository** and install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. **Generate the `labeled_outputs/` files**  
   You need to run your labeling model/script separately to create the required CSV files under a folder named `labeled_outputs/`.

3. **Fine-tuned BERT Model**
   - The code expects a fine-tuned BERT model named `"fine_tuned_bert_emotions"`.
   - Either:
     - Fine-tune BERT yourself using an emotions dataset (such as GoEmotions) using the emotions_bert_fine_tuned.py file,
     - Or use the uploaded hugging face model (default implemented) 'sdeakin/fine_tuned_bert_emotions' (i just tried this and its not working with this and I'm trying to fix it, but the locale file and folder definitely work)

4. **Run the Classification Pipeline**
    ```bash
    python classification_pipeline.py
    ```

   This will:
   - Load the labeled data
   - Generate BERT and TF-IDF features
   - Train and evaluate all selected models
   - Save model comparisons and plots

---

## Datasets Used

The pipeline is designed to run across six datasets + emotions:

| Dataset | Description |
|:--------|:------------|
| `labeled_triggers_keywords.csv` | Trigger labels based on keyword matching |
| `labeled_triggers_sentiment.csv` | Trigger labels based on sentiment analysis |
| `labeled_themes_keywords.csv` | Theme labels based on keyword matching |
| `labeled_themes_sentiment.csv` | Theme labels based on sentiment analysis |
| `labeled_symptoms_keywords.csv` | Symptom labels based on keyword matching |
| `labeled_symptoms_sentiment.csv` | Symptom labels based on sentiment analysis |
| `go_emotions_dataset.csv` | Taken from kaggle, Google's goEmotions database |

You can comment out datasets you are not using by modifying `datasets = [...]` in `classification_pipeline.py`.

---

## Outputs

After training, the following outputs will be generated:

- `results/final_model_results_<dataset>.csv`: Consolidated model performance metrics
- `plots/<dataset>/all_models_training_plot.png`: Overall training curves for all models
- `plots/grouped/<dataset>/*.png`: Training curves grouped by model type (CNN, MLP, etc.)
- `logs/`: Per-class precision, recall, and F1 score CSVs for each model

---

## Acknowledgments

This project uses the following libraries and models:
- HuggingFace Transformers
- Scikit-learn
- PyTorch
- Colorcet (for colorful training plots)
- GoEmotions Dataset
