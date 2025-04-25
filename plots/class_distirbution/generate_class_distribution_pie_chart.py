import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_class_distribution_from_file(csv_path, label_column, title, output_name):
    # Load labeled CSV
    df = pd.read_csv(csv_path)
    print("Available columns:", df.columns.tolist())

    # Drop missing or invalid labels
    df = df[df[label_column].notna()]
    
    # Count label occurrences
    label_counts = df[label_column].value_counts()

    # Custom autopct function to show % and count
    def make_autopct(values):
        def autopct(pct):
            total = sum(values)
            count = int(round(pct * total / 100.0))
            return f"{pct:.1f}%\n({count})"
        return autopct

    # Plot pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        label_counts.values,
        labels=label_counts.index,
        autopct=make_autopct(label_counts.values),
        startangle=140,
        textprops={'fontsize': 9}
    )
    plt.title(title)
    plt.axis('equal')

    # Save to file
    os.makedirs("plots", exist_ok=True)
    save_path = os.path.join("plots", f"pie_{output_name}.png")
    plt.savefig(save_path)
    print(f"📊 Pie chart saved: {save_path}")
    plt.show()

# Run for all
plot_class_distribution_from_file("labeled_outputs/labeled_triggers.csv", "triggers_label", "Trigger Category Distribution", "triggers")
plot_class_distribution_from_file("labeled_outputs/labeled_themes.csv", "themes_label", "Theme Category Distribution", "themes")
plot_class_distribution_from_file("labeled_outputs/labeled_symptoms.csv", "symptoms_label", "Symptom Category Distribution", "symptoms")
plot_class_distribution_from_file("labeled_outputs/labeled_emotions.csv", "emotion_label", "Emotion Category Distribution", "emotions")

