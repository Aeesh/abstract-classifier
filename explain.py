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
def visualize_attributions(tokens, attributions, title="Token Attributions"):
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
    plt.savefig("token_attributions.png", dpi=150)
    print("Attribution plot saved to token_attributions.png")
    plt.show()

if __name__ == "__main__":

    sample_abstract = """
        We report a first-principles density functional theory
        study with Hubbard U corrections (DFT+U) on four binary
        tungstate systems of the wolframite family: (Mn,X)WO4 where
        X = Fe, Co, Ni, and Zn. The Hubbard U parameters were derived
        from linear response theory, providing a non-empirical correction
        to d-orbital self-interaction errors. Both ferromagnetic (FM) and
        antiferromagnetic (AFM) spin configurations were considered.
        All converged systems are indirect-gap semiconductors with band
        gaps ranging from 2.236 eV (MnFeWO4, FM) to 2.432 eV
        (MnZnWO4). The gap follows the trend Zn > Ni > Co > Fe,
        correlating with the d-shell filling of the X-site cation. The AFM
        configuration is the ground state for MnCoWO4 and MnNiWO4;
        MnZnWO4 is magnetically degenerate. Band gap differences
        between FM and AFM are less than 15 meV for all systems.
        Computed gaps are compared against experimental optical gaps
        of the single-metal end-member wolframites; the binary values
        lie between the MnWO4 parent (∼2.5 eV) and the X-site endmember, physically consistent with 50% A-site mixing. Band
        structure and density of states results agree to within 0.02 eV.
        Index Terms—DFT+U, wolframite, tungstate, band gap, antiferromagnetic, linear response, electronic structure
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
        title=f"Token Attributions → {result['predicted_class']}"
    )

    with open("explanation.json", "w") as f:
        json.dump(result, f)
    print("\nExplanation details saved to explanation.json")