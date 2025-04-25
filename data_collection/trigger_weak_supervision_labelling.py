import pandas as pd
import re

# Load triggers list
triggers_df = pd.read_csv("Triggers - Sheet1.csv")

# Create a dictionary mapping categories to triggers
trigger_dict = {}
for _, row in triggers_df.iterrows():
    category = row["Trigger Categories"]
    trigger = row["Triggers"]
    
    if category not in trigger_dict:
        trigger_dict[category] = []
    trigger_dict[category].append(trigger.lower())

# Load dataset
df = pd.read_csv("reddit_mental_health_data.csv")

# Convert text column to string and handle missing values
df["text"] = df["text"].astype(str).fillna("")

# **🔹 Updated Function to Capture Multiple Triggers**
def label_triggers(text):
    text = str(text).lower()  # Convert to lowercase and ensure it's a string
    matched_categories = set()

    for category, triggers in trigger_dict.items():
        # If ANY trigger in the category is found, add the category
        if any(re.search(rf'\b{re.escape(trigger)}\b', text) for trigger in triggers):
            matched_categories.add(category)  # Store all matching categories

    return list(matched_categories) if matched_categories else ["No Trigger Found"]

# Apply weak supervision to label dataset
df["trigger_label"] = df["text"].apply(label_triggers)

# **Now, this captures multiple triggers per text entry!**

# Show labeled data
print(df[["text", "trigger_label"]].head())

# Filter dataset where trigger_label is not ["No Trigger Found"]
df_with_triggers = df[df["trigger_label"].apply(lambda x: x != ["No Trigger Found"])]

# Optionally, print the first few examples
print(df_with_triggers[["text", "trigger_label"]].head())

# Count how many rows contain at least one trigger
print(len(df_with_triggers))

# **Explode trigger_label column to count each category separately**
df_exploded = df_with_triggers.explode("trigger_label")

# Count occurrences of each trigger category
trigger_counts = df_exploded["trigger_label"].value_counts()

# Print the trigger counts
print("Trigger Category Counts:")
print(trigger_counts)

# **Save to CSV**
output_file = "labeled_triggers_exploded.csv"
df_exploded.to_csv(output_file, index=False)

print(f"Exploded labeled triggers saved to {output_file}")