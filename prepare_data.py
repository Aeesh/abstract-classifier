from datasets import load_dataset
import pandas as pd
import json
import os


# map labels from HF datasetvb u
ID2LABEL = {
    0: "math.AC — Commutative Algebra",
    1: "cs.CV — Computer Vision",
    2: "cs.AI — Artificial Intelligence",
    3: "cs.SY — Systems and Control",
    4: "math.GR — Group Theory",
    5: "cs.CE — Computational Engineering",
    6: "cs.PL — Programming Languages",
    7: "cs.IT — Information Theory",
    8: "cs.DS — Data Structures & Algorithms",
    9: "cs.NE — Neural & Evolutionary Computing",
    10: "math.ST — Statistics Theory"
}

LABEL2ID = {v: k for k, v in ID2LABEL.items()}

print("loading dataset...")
dataset = load_dataset("ccdv/arxiv-classification")

# convert dataset splits to pandas dataframe
train_df = pd.DataFrame(dataset["train"])
val_df = pd.DataFrame(dataset["validation"])
test_df = pd.DataFrame(dataset["test"])

# rename columns
train_df = train_df.rename(columns={'text': 'abstract'})
val_df = val_df.rename(columns={'text': 'abstract'})
test_df = test_df.rename(columns={'text': 'abstract'})

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print("\nClass distribution in train set:")
print(train_df['label'].value_counts().sort_index())

# save dataframes to csv files
os.makedirs("data", exist_ok=True)
train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

# save label mappings to json file
with open("data/label_map.json", "w") as f:
  json.dump({
    "id2label": {str(k): v for k, v in ID2LABEL.items()},
    "label2id": {str(k): k for k in range(11)},
    "num_labels": 11
  }, f, indent=2)

print("\nData saved to data/")
print("Label map saved to data/label_map.json")