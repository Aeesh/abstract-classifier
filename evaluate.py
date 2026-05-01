import torch
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

from utils import AbstractDataset


SAVED_DIR = "saved_model"
MAX_LENGTH = 512
BATCH_SIZE = 16

with open("data/label_map.json") as f:
    maps = json.load(f)

id2label = {int(k): v for k, v in maps["ids2label"].items()}
num_labels = maps["num_labels"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = DistilBertTokenizer.from_pretrained(SAVED_DIR)
model = DistilBertForSequenceClassification.from_pretrained(SAVED_DIR)
model = model.to(device)
model.eval()

test_dataset = AbstractDataset("data/test.csv", tokenizer, MAX_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        outputs = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device)
        )
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['label'].numpy())

label_names = [id2label[i] for i in range(num_labels)]

accuracy = accuracy_score(all_labels, all_preds)
macro_f1 = f1_score(all_labels, all_preds, average='macro')

print(f"\nTest Accuracy : {acc:.4f}")
print(f"Macro F1      : {macro_f1:.4f}")
print(f"\nPer-class report:")
print(classification_report(all_labels, all_preds, target_names=label_names))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 9))
sns.heatmap(
  cm, annot=True, fmt='d', cmap='Blues',
  xticklabels=label_names, yticklabels=label_names
)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix — arXiv Abstract Classifier', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nConfusion matrix saved to confusion_matrix.png")

with open("test_metrics.json", "w") as f:
    json.dump({
        "test_accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": classification_report(
            all_labels, all_preds,
            target_names=label_names,
            output_dict=True
        )
    }, f, indent=2)
print("Metrics saved to test_metrics.json")