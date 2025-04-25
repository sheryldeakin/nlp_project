import pandas as pd

# Load GoEmotions
emotions_df = pd.read_csv("data/go_emotions_dataset.csv")

# Show all columns to debug
print("All columns in dataset:", emotions_df.columns.tolist())

# Filter only numeric (label) columns
emotion_columns = [
    col for col in emotions_df.columns 
    if col not in ["text", "id", "example_very_unclear", "example_id", "example_url"] 
    and emotions_df[col].dropna().apply(lambda x: str(x).isdigit()).all()
]

print("Detected emotion label columns:", emotion_columns)

# Convert to int
emotions_df[emotion_columns] = emotions_df[emotion_columns].astype(int)

# Create multi-label column
emotions_df["emotion_label"] = emotions_df[emotion_columns].apply(
    lambda row: [emotion for emotion, val in row.items() if val == 1], axis=1
)

# Filter out rows with no labels
emotions_labeled = emotions_df[emotions_df["emotion_label"].apply(lambda x: len(x) > 0)].copy()

# Explode for per-label pie chart
emotions_exploded = emotions_labeled.explode("emotion_label")

# Save
emotions_exploded.to_csv("labeled_emotions.csv", index=False)
print("✅ Saved: labeled_emotions.csv")
