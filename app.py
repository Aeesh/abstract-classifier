import torch
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from captum.attr import IntegratedGradients

SAVE_DIR = "saved_model"
MAX_LENGTH = 512

st.set_page_config(
    page_title="arXiv Abstract Classifier",
    page_icon="📄",
    layout="wide"
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 arXiv Classifier")
    st.markdown(
        "Fine-tuned **DistilBERT** model that classifies scientific abstracts "
        "into 11 arXiv fields."
    )
    st.markdown("---")
    st.markdown("**Model details**")
    st.markdown("- Base: `distilbert-base-uncased`")
    st.markdown("- Parameters: 66.9M")
    st.markdown("- Test accuracy: **88%**")
    st.markdown("- Macro F1: **0.877**")
    st.markdown("- Trained on PSC Bridges-2 V100")
    st.markdown("---")
    st.markdown("**Links**")
    st.markdown("[🤗 Model on HuggingFace](https://huggingface.co/aeesh1/arxiv-abstract-classifier)")
    st.markdown("[💻 GitHub](https://github.com/aeesh/abstract-classifier.git)")
    st.markdown("[📊 WandB Run](https://wandb.ai/aeesh-aeesh/arxiv-abstract-classifier/runs/l5quax1q)")
    st.markdown("---")
    run_explainability = st.checkbox("Show token attributions", value=True)
    st.caption(
        "Attributions use integrated gradients (Captum). "
        "Green = pushed toward prediction. Red = pushed against."
    )

# ── Example abstracts ────────────────────────────────────────────────────────
EXAMPLES = {
    "Select an example...": "",
    "Attention Is All You Need (cs.NE)": (
        "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks "
        "in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through "
        "an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention "
        "mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these "
        "models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model "
        "achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including "
        "ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art "
        "BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. "
        "We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data."
    ),
    "Monkeypox Detection with ConvNeXt (cs.CV / cs.AI boundary)": (
        "Monkeypox (Mpox) is an infectious disease caused by a virus belonging to the same family as the smallpox virus, "
        "presenting significant diagnostic challenges due to its visual similarity to other skin conditions like chickenpox. "
        "While PCR testing remains the gold standard for diagnosis, it is often inaccessible in resource-limited settings, necessitating "
        "the development of rapid and reliable alternative detection methods [1]. In this work, we extend the study by Ahsan et al. "
        "[2] which proposed a modified VGG16 model for Monkeypox detection, achieving an accuracy of 0.97 ± 0.018% (AUC = 97.2) "
        "in their primary (study one) experiment. To enhance this baseline, we implemented several key improvements: (1) replacing VGG16 with the modern ConvNeXt architecture "
        "to leverage its transformer-inspired design for better feature extraction, (2) applying advanced hyperparameter optimization techniques, and (3) incorporating explainable AI "
        "methods like Grad-CAM to improve model interpretability. Our enhanced model achieved a peak accuracy of 88.89% (precision: 1.0000, recall: "
        "0.7778, F1-score: 0.8750) on the test set, surpassing the original study’s test accuracy of "
        "83% ± 0.085. Notably, the model demonstrated robust specificity (1.0000) and a test AUC "
        "of 0.89, highlighting its potential for clinical deployment. Despite challenges posed by the "
        "limited dataset size, our improvements in architecture and optimization underscore the "
        "viability of deep learning for Monkeypox detection, particularly in resource-constrained "
        "regions like East Africa, where recent outbreaks have been reported "
    ),
    "DFT study — out-of-distribution (materials physics)": (
        "We report a first-principles density functional theory "
        "study with Hubbard U corrections (DFT+U) on four binary "
        "tungstate systems of the wolframite family: (Mn,X)WO4 where "
        "X = Fe, Co, Ni, and Zn. The Hubbard U parameters were derived "
        "from linear response theory, providing a non-empirical correction "
        "to d-orbital self-interaction errors. Both ferromagnetic (FM) and "
        "antiferromagnetic (AFM) spin configurations were considered. "
        "All converged systems are indirect-gap semiconductors with band "
        "gaps ranging from 2.236 eV (MnFeWO4, FM) to 2.432 eV "
        "(MnZnWO4). The gap follows the trend Zn > Ni > Co > Fe, "
        "correlating with the d-shell filling of the X-site cation. The AFM "
        "configuration is the ground state for MnCoWO4 and MnNiWO4; "
        "MnZnWO4 is magnetically degenerate. Band gap differences "
        "between FM and AFM are less than 15 meV for all systems. "
        "Computed gaps are compared against experimental optical gaps "
        "of the single-metal end-member wolframites; the binary values "
        "lie between the MnWO4 parent (∼2.5 eV) and the X-site endmember, physically consistent with 50% A-site mixing. Band "
        "structure and density of states results agree to within 0.02 eV. "
        "Index Terms—DFT+U, wolframite, tungstate, band gap, antiferromagnetic, linear response, electronic structure "
    ),
}

# ── Load model ───────────────────────────────────────────────────────────────
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

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("arXiv Abstract Field Classifier")
st.markdown(
    "Paste any scientific abstract to classify it into one of 11 arXiv fields. The model predicts the arXiv field "
    "and highlights which words drove the prediction."
)


# Example selector
selected_example = st.selectbox("Or load an example:", list(EXAMPLES.keys()))
example_text = EXAMPLES[selected_example]

abstract = st.text_area(
    "Abstract:",
    value=example_text,
    height=180,
    placeholder="Paste your abstract here..."
)

classify_btn = st.button("Classify", type="primary")

if classify_btn and abstract.strip():

    with st.spinner("Classifying..."):
        encoding = tokenizer(
            abstract,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).logits
            probs = torch.softmax(logits, dim=1).squeeze()
            pred_id = torch.argmax(probs).item()

    confidence = probs[pred_id].item()
    predicted_label = id2label[pred_id]

    # Confidence colour
    if confidence >= 0.75:
        badge_color = "green"
        conf_note = ""
    elif confidence >= 0.50:
        badge_color = "orange"
        conf_note = "Moderate confidence — prediction may be unreliable."
    else:
        badge_color = "red"
        conf_note = (
            "Low confidence — this abstract may be outside the model's training distribution. "
            "The model only knows 11 arXiv categories."
        )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### Prediction")
        st.markdown(
            f"<h2 style='color:{badge_color}'>{predicted_label}</h2>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<h4 style='color:{badge_color}'>Confidence: {confidence:.1%}</h4>",
            unsafe_allow_html=True
        )
        if conf_note:
            st.warning(conf_note)

        st.markdown("---")
        st.markdown("**Top 3 predictions**")

        scores = [(id2label[i], probs[i].item()) for i in range(len(id2label))]
        top3 = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
        for rank, (label, score) in enumerate(top3):
            icon = "🥇" if rank == 0 else ("🥈" if rank == 1 else "🥉")
            st.markdown(f"{icon} **{label}** — {score:.1%}")

        st.markdown("---")
        with st.expander("All class probabilities"):
            all_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
            for label, score in all_sorted:
                st.progress(score, text=f"{label}: {score:.1%}")

    with col2:
        if run_explainability:
            st.markdown("### Token Attributions")
            st.caption("Top 25 most influential tokens. Green = toward prediction. Red = against.")

            with st.spinner("Computing integrated gradients..."):
                embeddings = model.distilbert.embeddings(input_ids)
                baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
                baseline_embeddings = model.distilbert.embeddings(baseline_ids)

                def predict_proba(embeds):
                    return model(inputs_embeds=embeds).logits

                ig = IntegratedGradients(predict_proba)
                attributions, _ = ig.attribute(
                    embeddings,
                    baseline_embeddings,
                    target=pred_id,
                    n_steps=30,
                    return_convergence_delta=True
                )

            attributions = attributions.sum(dim=-1).squeeze(0)
            attributions = attributions / torch.norm(attributions)
            attributions = attributions.detach().numpy()

            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].numpy())
            real_length = attention_mask[0].sum().item()
            tokens = tokens[1:real_length - 1]
            attributions = attributions[1:real_length - 1]

            abs_attrs = np.abs(attributions)
            top_idx = np.argsort(abs_attrs)[-25:][::-1]
            top_tokens = [tokens[i] for i in top_idx]
            top_attrs = [attributions[i] for i in top_idx]
            colors = ["#2ecc71" if a > 0 else "#e74c3c" for a in top_attrs]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(range(len(top_tokens)), top_attrs, color=colors)
            ax.set_xticks(range(len(top_tokens)))
            ax.set_xticklabels(top_tokens, rotation=45, ha="right", fontsize=8)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_title(f"Token Attributions → {predicted_label}", fontsize=11)
            ax.set_ylabel("Attribution Score", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)

st.markdown("---")
st.caption(
    "Model: fine-tuned distilbert-base-uncased · Dataset: ccdv/arxiv-classification · "
    "Explainability: Captum integrated gradients · Trained on PSC Bridges-2 V100"
)