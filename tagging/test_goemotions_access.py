#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick check: can we load and access the GoEmotions dataset?
This will print a few sample rows and counts.
"""

from datasets import load_dataset

def test_goemotions_access(limit=5):
    print("🔍 Loading GoEmotions (simplified split) from Hugging Face...")
    try:
        ds = load_dataset("go_emotions", "simplified")
    except Exception as e:
        print("❌ Could not load dataset:", e)
        print("\n👉 Try running:  pip install datasets\n"
              "or check your internet connection / Hugging Face access.")
        return

    print("\n✅ Dataset splits available:", list(ds.keys()))

    for split in ds:
        print(f"📦 {split}: {len(ds[split])} examples")

    # show a few examples
    split = "train" if "train" in ds else list(ds.keys())[0]
    print(f"\n🧩 Showing first {limit} examples from '{split}':\n")

    for i, ex in enumerate(ds[split].select(range(limit))):
        print(f"[{i}] id={ex.get('id', 'NA')} | text={ex['text']}")
        if "labels" in ex:
            print("   labels:", ex["labels"])
        print()

    print("\n✅ GoEmotions dataset loaded successfully!")

if __name__ == "__main__":
    test_goemotions_access()