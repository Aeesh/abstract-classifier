import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from captum.attr import IntegratedGradients

SAVE_DIR = "saved_model"

with open("data/label_map.json") as f:
    maps = json.load(f)
id2label = {int(k): v for k, v in maps["id2label"].items()}

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained(SAVE_DIR)
model = DistilBertForSequenceClassification.from_pretrained(SAVE_DIR)
model = model.to(device)
model.eval()

# Wrapper that takes input_ids embeddings and returns logits.
def predict_probability(input_ids):
    outputs = model(inputs_embeds=input_ids)
    return outputs.logits

# Given a text string, return the predicted class and token attributions.
def explain_prediction(text, max_length=512):
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Get token embeddings
    embeddings = model.distilbert.embeddings(input_ids)

    # Baseline: all paddimng token embeddings
    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
    baseline_embeddings = model.distilbert.embeddings(baseline_ids)

    # Get prediction
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        pred_id = torch.argmax(logits, dim=1).item()
        probs = torch.softmax(logits, dim=1).squeeze()

    # Run Integrated Gradients
    ig = IntegratedGradients(predict_probability)
    attributions, delta = ig.attribute(
        embeddings,
        baseline_embeddings,
        target=pred_id,
        n_steps=50,
        return_convergence_delta=True
    )

