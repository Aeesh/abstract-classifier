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
def forward_fn(input_ids):
    outputs = model(inputs_embeds=input_ids)
    return outputs.logits

# Given a text string, return the predicted class and token attributions.
def explain_prediction(text, max_length=512):
    print("Starting explanation...")
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
    print("Running IG...")
    ig = IntegratedGradients(forward_fn)
    attributions, delta = ig.attribute(
        embeddings,
        baseline_embeddings,
        target=pred_id,
        n_steps=50,
        return_convergence_delta=True
    )

    print("IG done. Now summing attributions...")

    # sum attributions across embedding dimensions to get per-token score
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)  # normalize for visualization
    attributions = attributions.cpu().detach().numpy()

    # Decode tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())

    # Kepp only non-padding tokens
    real_length = attention_mask[0].sum().item()
    tokens = tokens[:real_length]
    attributions = attributions[:real_length]

    return {
        "predicted_class": id2label[pred_id],
        "confidence": probs[pred_id].item(),
        "all_probs": {id2label[i]: probs[i].item() for i in range(len(id2label))},
        "tokens": tokens,
        "attributions": attributions.tolist()
    }


# visualization function to display tokens with color intensity based on attribution scores
def visualize_attributions(tokens, attributions, title="Token Attributions", text=""):
    # Get top 30 by absolute value
    abs_attrs = np.abs(attributions)
    top_indices = np.argsort(abs_attrs)[-30:][::-1]
    top_tokens = [tokens[i] for i in top_indices]
    top_attrs = [attributions[i] for i in top_indices]

    colors = ['#2ecc71' if a > 0 else '#e74c3c' for a in top_attrs]

    # Plot a bar chart of token attributions.
    # Positive = pushed toward prediction.
    # Negative = pushed away.
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_tokens)), top_attrs, color=colors)
    plt.xticks(range(len(top_tokens)), top_tokens, rotation=45, ha='right', fontsize=9)
    plt.axhline(y=0, color='black', linewidth=0.5)
    plt.title(title, fontsize=13)
    plt.ylabel("Attribution Score")
    plt.tight_layout()
    plt.savefig(f"token_attributions_{text}.png", dpi=150)
    print(f"Attribution plot saved to token_attributions_{text}.png")

if __name__ == "__main__":

    sample_abstract_name = "monkeypox"
    sample_abstract = """
        Monkeypox (Mpox) is an infectious disease caused by a virus belonging to the same family as the smallpox virus, presenting significant diagnostic challenges due to its visual
        similarity to other skin conditions like chickenpox. While PCR testing remains the gold
        standard for diagnosis, it is often inaccessible in resource-limited settings, necessitating
        the development of rapid and reliable alternative detection methods [1].
        In this work, we extend the study by Ahsan et al. [2] which proposed a modified VGG16
        model for Monkeypox detection, achieving an accuracy of 0.97 ± 0.018% (AUC = 97.2)
        in their primary (study one) experiment. To enhance this baseline, we implemented several key improvements: (1) replacing VGG16 with the modern ConvNeXt architecture
        to leverage its transformer-inspired design for better feature extraction, (2) applying advanced hyperparameter optimization techniques, and (3) incorporating explainable AI
        methods like Grad-CAM to improve model interpretability.
        Our enhanced model achieved a peak accuracy of 88.89% (precision: 1.0000, recall:
        0.7778, F1-score: 0.8750) on the test set, surpassing the original study’s test accuracy of
        83% ± 0.085. Notably, the model demonstrated robust specificity (1.0000) and a test AUC
        of 0.89, highlighting its potential for clinical deployment. Despite challenges posed by the
        limited dataset size, our improvements in architecture and optimization underscore the
        viability of deep learning for Monkeypox detection, particularly in resource-constrained
        regions like East Africa, where recent outbreaks have been reported
    """
    result = explain_prediction(sample_abstract)

    print(f"Predicted: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.1%}")

    print("\nAll class probabilities:")
    for cls, prob in sorted(result['all_probs'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls}: {prob:.1%}")

    print("Creating plot...")
    visualize_attributions(
        result['tokens'],
        result['attributions'],
        title=f"Token Attributions → {result['predicted_class']}",
        text=sample_abstract_name
    )

    with open(f"explanation_{sample_abstract_name}.json", "w") as f:
        json.dump(result, f)
    print(f"\nExplanation details saved to explanation_{sample_abstract_name}.json")