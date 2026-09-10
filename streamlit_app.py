"""
Streamlit Web Application: Layer-wise Relevance Propagation Explorer
=====================================================================
Run with: streamlit run streamlit_app.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch

from lrp_engine import LRPEngine
from tabular_lrp import build_model as build_tabular_model
from image_lrp import build_convnet as build_image_model


st.set_page_config(
    page_title="LRP Explainability Studio",
    page_icon="🔍",
    layout="wide",
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


@st.cache_resource
def load_resources():
    # 1. Metadata
    with open(os.path.join(MODELS_DIR, "metadata.json"), "r") as f:
        meta = json.load(f)
    with open(os.path.join(MODELS_DIR, "sample_bank.json"), "r") as f:
        sample_bank = json.load(f)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    # 2. Tabular model
    tab_m = build_tabular_model(input_dim=30)
    tab_m.load_state_dict(torch.load(os.path.join(MODELS_DIR, "tabular_mlp.pth"), map_location="cpu"))
    tab_m.eval()
    tab_engine = LRPEngine(tab_m)

    # 3. Image model
    img_m = build_image_model()
    img_m.load_state_dict(torch.load(os.path.join(MODELS_DIR, "image_cnn.pth"), map_location="cpu"))
    img_m.eval()
    img_engine = LRPEngine(img_m)

    return meta, sample_bank, scaler, tab_m, tab_engine, img_m, img_engine


meta, sample_bank, scaler, tab_m, tab_engine, img_m, img_engine = load_resources()

st.title("🔍 Layer-wise Relevance Propagation (LRP) Studio")
st.markdown(
    "Explainable AI (XAI) analysis using the **Deep Taylor Decomposition** framework. "
    "Verify the layer-wise conservation law $\\sum R_i = f(x)_c$ and analyze feature attributions."
)

tab1, tab2 = st.tabs(["📊 Tabular Biomarker Attribution (MLP)", "🖼️ Visual Pixel Heatmaps (ConvNet)"])

# -----------------------------------------------------------------------------
# TAB 1: Tabular LRP
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Breast Cancer Wisconsin Diagnostic Classifier")
    col_ctrl, col_main = st.columns([1, 2.5])

    with col_ctrl:
        st.markdown("### Controls")
        sample_labels = [f"Sample #{s['id']} ({s['label_name'].upper()})" for s in sample_bank["tabular_samples"]]
        sample_choice = st.selectbox("Select Patient Case", range(len(sample_labels)), format_func=lambda i: sample_labels[i])
        selected_sample = sample_bank["tabular_samples"][sample_choice]

        rule = st.selectbox("LRP Rule", ["lrp-0", "lrp-epsilon", "lrp-gamma"])
        eps = 1e-4
        gamma = 0.25
        if rule == "lrp-epsilon":
            eps = st.slider("Epsilon (Noise Filter)", 0.0001, 0.5, 0.05, 0.01)
        elif rule == "lrp-gamma":
            gamma = st.slider("Gamma (Positive Multiplier)", 0.05, 1.0, 0.25, 0.05)

        top_k = st.slider("Top Features to Display", 5, 30, 15)

    with col_main:
        x_raw = np.array(selected_sample["raw_features"])
        x_scaled = scaler.transform(x_raw.reshape(1, -1))[0]
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = tab_m(x_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze().numpy()

        exp = tab_engine.explain(x_tensor, rule=rule, epsilon=eps, gamma=gamma)
        pred_class = exp["predicted_class"]
        pred_name = meta["tabular"]["target_names"][pred_class]
        target_score = exp["target_score"]
        sum_r = exp["sum_input_relevance"]
        error = exp["conservation_error"]

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Diagnosis", pred_name.upper(), f"{probs[pred_class]*100:.1f}% confidence")
        m2.metric("Target Logit f(x)", f"{target_score:.3f}")
        m3.metric("Relevance Sum ∑R", f"{sum_r:.3f}")
        m4.metric("Conservation Error", f"{error:.2e}")

        # Feature table & bar plot
        relevances = exp["input_relevance"].squeeze().numpy()
        feat_names = meta["tabular"]["feature_names"]
        df_feats = pd.DataFrame({
            "Feature": feat_names,
            "Relevance": relevances,
            "Raw Value": x_raw,
            "Absolute Relevance": np.abs(relevances)
        }).sort_values(by="Absolute Relevance", ascending=False).head(top_k)

        # Matplotlib Horizontal Bar Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#3b82f6" if r < 0 else "#ef4444" for r in df_feats["Relevance"][::-1]]
        bars = ax.barh(df_feats["Feature"][::-1], df_feats["Relevance"][::-1], color=colors, edgecolor="black", alpha=0.85)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Relevance Score (Red = Supports, Blue = Opposes)")
        ax.set_title(f"LRP Feature Attribution (Rule: {rule})")
        plt.tight_layout()
        st.pyplot(fig)

        # Download CSV
        csv = df_feats.to_csv(index=False)
        st.download_button("📥 Download Feature Attributions (CSV)", data=csv, file_name=f"lrp_attributions_sample_{selected_sample['id']}.csv")

# -----------------------------------------------------------------------------
# TAB 2: Image LRP
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Handwritten Digits Convolutional Classifier")
    col_img_ctrl, col_img_main = st.columns([1, 2.5])

    with col_img_ctrl:
        digit_choice = st.selectbox("Select Digit", list(range(10)), index=3)
        img_rule = st.selectbox("LRP Rule (Image)", ["lrp-epsilon", "lrp-0", "lrp-gamma"], key="img_rule")
        comp_choice = st.selectbox("Contrastive Class (Competitor)", [8, 5, 7, 2, 0], index=0)

    with col_img_main:
        sample_img = next(s for s in sample_bank["image_samples"] if s["digit"] == digit_choice)
        pixels = np.array(sample_img["pixels_8x8"])
        x_img_tensor = torch.tensor(pixels, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            logits_img = img_m(x_img_tensor)
            probs_img = torch.softmax(logits_img, dim=-1).squeeze().numpy()

        exp_pred = img_engine.explain(x_img_tensor, rule=img_rule)
        exp_comp = img_engine.explain(x_img_tensor, target_class=comp_choice, rule=img_rule)

        im1, im2, im3, im4 = st.columns(4)
        im1.metric("Predicted Digit", f"Digit {exp_pred['predicted_class']}", f"{probs_img[exp_pred['predicted_class']]*100:.1f}%")
        im2.metric("Target Logit", f"{exp_pred['target_score']:.2f}")
        im3.metric("Relevance Sum", f"{exp_pred['sum_input_relevance']:.2f}")
        im4.metric("Conservation Error", f"{exp_pred['conservation_error']:.2e}")

        # Heatmaps row
        fig_h, axes = plt.subplots(1, 3, figsize=(12, 4))
        rel_p = exp_pred["input_relevance"].squeeze().numpy()
        rel_c = exp_comp["input_relevance"].squeeze().numpy()
        vmax_p = max(np.max(np.abs(rel_p)), 1e-5)
        vmax_c = max(np.max(np.abs(rel_c)), 1e-5)

        axes[0].imshow(pixels, cmap="gray_r")
        axes[0].set_title(f"Input Digit ({digit_choice})")
        axes[0].axis("off")

        im_p = axes[1].imshow(rel_p, cmap="seismic", vmin=-vmax_p, vmax=vmax_p)
        axes[1].set_title(f"Relevance for Class '{exp_pred['predicted_class']}' (Predicted)")
        axes[1].axis("off")
        plt.colorbar(im_p, ax=axes[1], fraction=0.046)

        im_c = axes[2].imshow(rel_c, cmap="seismic", vmin=-vmax_c, vmax=vmax_c)
        axes[2].set_title(f"Relevance for Class '{comp_choice}' (Competitor)")
        axes[2].axis("off")
        plt.colorbar(im_c, ax=axes[2], fraction=0.046)

        plt.tight_layout()
        st.pyplot(fig_h)
