import json
import torch
import pandas as pd
from torch.utils.data import Dataset


class AbstractDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['abstract'])
        label = int(row['label'])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_label_map(path="data/label_map.json"):
    """Load id2label and label2id from the saved JSON."""
    with open(path) as f:
        maps = json.load(f)
    id2label = {int(k): v for k, v in maps["id2label"].items()}
    label2id = {v: int(k) for k, v in maps["id2label"].items()}
    num_labels = maps["num_labels"]
    return id2label, label2id, num_labels