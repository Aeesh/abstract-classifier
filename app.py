import torch
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from captum.attr import IntegratedGradients

SAVE_DIR = "saved_model"
MAX_LENGTH = 512

st.set_page_config(page_title="arXiv Abstract Classifier", layout="wide")
st.title("arXiv Abstract Field Classifier")
st.write(
    "Paste any scientific abstract. The model predicts the arXiv field "
    "and highlights which words drove the prediction."
)

@st.cache_resource
def load_model():
    with open("data/label_map.json") as f:
        maps = json.load(f)
    id2label = {int(k): v for k, v in maps["id2label"].items()}

    tokenizer = DistilBertTokenizer.from_pretrained(SAVE_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(SAVE_DIR)
    model.eval()
    return tokenizer, model, id2label

tokenizer, model, id2label = load_model()
device = torch.device("cpu")

abstract = st.text_area("Abstract", height=200, placeholder="Paste your abstract here...")

run_explainability = st.checkbox("Show token attributions (slower)", value=True)

if st.button("Classify") and abstract.strip():
    with st.spinner("Running..."):
        encoding = tokenizer(
            abstract, max_length=MAX_LENGTH,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=1).squeeze()
            pred_id = torch.argmax(probs).item()


    col1, col2 = st.columns(2) # Split the page into two columns

    with col1:
        st.markdown(f"### Predicted: **{id2label[pred_id]}**")
        st.markdown(f"Confidence: **{probs[pred_id].item():.1%}**")

        st.markdown("#### All class scores")
        scores = {id2label[i]: probs[i].item() for i in range(len(id2label))}
        for field, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            st.progress(score, text=f"{field}: {score:.1%}")

    with col2:
        if run_explainability:
            st.markdown("#### Token attributions")
            st.caption("Green = pushed toward prediction. Red = pushed away.")

            embeddings = model.distilbert.embeddings(input_ids)
            baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
            baseline_embeddings = model.distilbert.embeddings(baseline_ids)

            def forward_func(embeds):
                return model(inputs_embeds=embeds).logits

            ig = IntegratedGradients(predict_proba)
            attributions, _ = ig.attribute(
                embeddings, baseline_embeddings,
                target=pred_id, n_steps=30, return_convergence_delta=True
            )
            attributions = attributions.sum(dim=-1).squeeze(0)
            attributions = attributions / torch.norm(attributions)
            attributions = attributions.detach().numpy()

            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].numpy())
            real_length = attention_mask[0].sum().item()
            tokens = tokens[1:real_length-1]       # strip [CLS] and [SEP]
            attributions = attributions[1:real_length-1]

            abs_attrs = np.abs(attributions)
            top_indices = np.argsort(abs_attrs)[-25:][::-1]
            top_tokens = [tokens[i] for i in top_indices]
            top_attrs = [attributions[i] for i in top_indices]

            colors = ['#2ecc71' if a > 0 else '#e74c3c' for a in top_attrs]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(range(len(top_tokens)), top_attrs, color=colors)
            ax.set_xticks(range(len(top_tokens)))
            ax.set_xticklabels(top_tokens, rotation=45, ha='right', fontsize=8)
            ax.axhline(0, color='black', linewidth=0.5)
            ax.set_title("Top 25 Most Influential Tokens")
            ax.set_ylabel("Attribution Score")
            plt.tight_layout()
            st.pyplot(fig)
