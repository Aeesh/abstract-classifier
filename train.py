import torch
import pandas as pd
import json
import numpy as np
import wandb
import os

from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)

from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from util import AbstractDataset


####### config #######
CONFIG = {
    "train_data_dir": "data/train.csv",
    "val_data_dir": "data/val.csv",
    "test_data_dir": "data/test.csv",
    "model_name": "distilbert-base-uncased",
    "max_length": 512,
    "batch_size": 16,
    "epochs": 6,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,     # 10% of steps used for LR warmup
    "weight_decay": 0.01,
    "save_dir": "saved_model",
    "num_workers": 2,
    "wandb_project": "arxiv-abstract-classifier"
}

# Initialise WandB
wandb.init(
    project=CONFIG["wandb_project"],
    config=CONFIG,
    name=f"distilbert-{CONFIG['epochs']}epochs"
)

# Load label map
with open("data/label_map.json") as f:
    maps = json.load(f)
id2label = {int(k): v for k, v in maps["id2label"].items()}
num_labels = maps["num_labels"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


####### Load tokenizer and model #######
print("Loading tokenizer and model...")
tokenizer = DistilBertTokenizer.from_pretrained(CONFIG["model_name"])

model = DistilBertForSequenceClassification.from_pretrained(
    CONFIG["model_name"],
    num_labels=num_labels,
    id2label=id2label,
    label2id={v: k for k, v in id2label.items()}
)
model.to(device)

####### log model parameters to WandB #######
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params} | Trainable parameters: {trainable_params}")
wandb.config.update({"total_params": total_params})


######## Data loaders #######
train_dataset = AbstractDataset(CONFIG["train_data_dir"], tokenizer, CONFIG["max_length"])
val_dataset = AbstractDataset(CONFIG["val_data_dir"], tokenizer, CONFIG["max_length"])

# # TODO: Remove this after testing
# train_dataset.df = train_dataset.df.head(64)
# val_dataset.df = val_dataset.df.head(32)

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

######## Class weights for imbalanced classes #######
train_labels = train_dataset.df['label'].values
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(num_labels),
    y=train_labels
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
print(f"class weights: {class_weights.round(3)}")


######## Optimizer and scheduler #######
optimizer = AdamW(
    model.parameters(),
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"]
)
total_steps = len(train_loader) * CONFIG["epochs"]
warmup_steps = int(total_steps * CONFIG["warmup_ratio"])

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

########  Evaluation function1 #######
def evaluate(model, loader, device, loss_fn):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, accuracy, macro_f1

######## Training loop #######
best_val_f1 = 0
print("\nStarting training...\n")

for epoch in range(CONFIG["epochs"]):
    model.train()
    total_train_loss = 0

    for step, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        total_train_loss += loss.item()

        # Log batch loss to WandB every 50 steps
        if step % 50 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(
              f" Epoch {epoch+1} | Step {step}/{len(train_loader)}"
              f" | Loss: {loss.item():.4f} | LR: {current_lr:.2e}"
            )
            wandb.log({
              "batch_loss": loss.item(),
              "learning_rate": current_lr,
              "step": epoch * len(train_loader) + step
            })

    avg_train_loss = total_train_loss / len(train_loader)
    val_loss, val_accuracy, val_f1 = evaluate(model, val_loader, device, loss_fn)

    print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
    print(f"  Train loss : {avg_train_loss:.4f}")
    print(f"  Val loss   : {val_loss:.4f}")
    print(f"  Val acc    : {val_accuracy:.4f}")
    print(f"  Val macro F1: {val_f1:.4f}\n")

    # Log epoch metrics to WandB
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_f1
    })

    # Save the best model based on validation F1
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        os.makedirs(CONFIG["save_dir"], exist_ok=True)
        model.save_pretrained(CONFIG["save_dir"])
        tokenizer.save_pretrained(CONFIG["save_dir"])
        print(f" Best model saved (Val F1: {val_f1:.4f})")
        wandb.run.summary["best_val_f1"] = val_f1
        wandb.run.summary["best_epoch"] = epoch + 1

wandb.finish()
print(f"\nTraining complete. Best Val F1: {best_val_f1:.4f}")
print(f"Model saved to {CONFIG['save_dir']}/")