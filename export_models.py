"""
Model Serialization & Export Script
===================================
Trains and exports production-ready artifacts:
  - models/tabular_mlp.pth (Trained PyTorch weights for Breast Cancer MLP)
  - models/scaler.pkl (Fitted StandardScaler for tabular inputs)
  - models/image_cnn.pth (Trained PyTorch weights for Digit Recognition CNN)
  - models/sample_bank.json (Pre-packaged test samples with ground truth for instant UI exploration)
  - models/metadata.json (Model architecture, features, and target class metadata)
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tabular_lrp import build_model as build_tabular_model, train_model as train_tabular_model, evaluate_model as eval_tabular
from image_lrp import build_convnet as build_image_model, train_convnet as train_image_model, evaluate_convnet as eval_image


def export_all(models_dir: str = "models"):
    os.makedirs(models_dir, exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("  SERIALIZING PRODUCTION MODEL ARTIFACTS")
    print("=" * 60)

    # -------------------------------------------------------------
    # 1. Tabular Model & Scaler (Breast Cancer Wisconsin)
    # -------------------------------------------------------------
    print("[1/3] Training and exporting Tabular MLP model...")
    cancer_data = load_breast_cancer()
    X_tab = cancer_data.data
    y_tab = cancer_data.target
    feature_names = cancer_data.feature_names.tolist()
    target_names = cancer_data.target_names.tolist()

    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        X_tab, y_tab, test_size=0.2, random_state=42, stratify=y_tab
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_t)
    X_test_scaled = scaler.transform(X_test_t)

    from torch.utils.data import TensorDataset
    train_ds_t = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train_t, dtype=torch.long),
    )
    test_ds_t = TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(y_test_t, dtype=torch.long),
    )

    tab_model = build_tabular_model(input_dim=len(feature_names))
    tab_model = train_tabular_model(tab_model, train_ds_t, epochs=70)
    tab_acc = eval_tabular(tab_model, test_ds_t)
    print(f"  -> Tabular MLP Test Accuracy: {tab_acc * 100:.2f}%")

    tab_model_path = os.path.join(models_dir, "tabular_mlp.pth")
    torch.save(tab_model.state_dict(), tab_model_path)
    print(f"  -> Saved weights: {tab_model_path}")

    scaler_path = os.path.join(models_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  -> Saved scaler: {scaler_path}")

    # -------------------------------------------------------------
    # 2. Image Model (Handwritten Digits CNN)
    # -------------------------------------------------------------
    print("\n[2/3] Training and exporting Image CNN model...")
    digits_data = load_digits()
    X_img = np.expand_dims(digits_data.images / 16.0, axis=1)  # (1797, 1, 8, 8)
    y_img = digits_data.target

    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
        X_img, y_img, test_size=0.2, random_state=42, stratify=y_img
    )

    train_ds_i = TensorDataset(
        torch.tensor(X_train_i, dtype=torch.float32),
        torch.tensor(y_train_i, dtype=torch.long),
    )
    test_ds_i = TensorDataset(
        torch.tensor(X_test_i, dtype=torch.float32),
        torch.tensor(y_test_i, dtype=torch.long),
    )

    img_model = build_image_model()
    img_model = train_image_model(img_model, train_ds_i, epochs=45)
    img_acc = eval_image(img_model, test_ds_i)
    print(f"  -> Image CNN Test Accuracy: {img_acc * 100:.2f}%")

    img_model_path = os.path.join(models_dir, "image_cnn.pth")
    torch.save(img_model.state_dict(), img_model_path)
    print(f"  -> Saved weights: {img_model_path}")

    # -------------------------------------------------------------
    # 3. Sample Bank & Metadata for Fast API & UI Exploration
    # -------------------------------------------------------------
    print("\n[3/3] Exporting sample bank and metadata...")
    # Curate 6 diverse tabular samples (3 malignant, 3 benign)
    mal_indices = np.where(y_test_t == 0)[0][:3]
    ben_indices = np.where(y_test_t == 1)[0][:3]

    tabular_samples = []
    for idx in list(mal_indices) + list(ben_indices):
        tabular_samples.append({
            "id": int(idx),
            "label": int(y_test_t[idx]),
            "label_name": target_names[y_test_t[idx]],
            "raw_features": X_test_t[idx].tolist(),
            "scaled_features": X_test_scaled[idx].tolist(),
        })

    # Curate 10 digit samples (one per class 0-9)
    image_samples = []
    for digit in range(10):
        digit_indices = np.where(y_test_i == digit)[0]
        if len(digit_indices) > 0:
            d_idx = digit_indices[0]
            image_samples.append({
                "id": int(d_idx),
                "digit": int(digit),
                "pixels_8x8": X_test_i[d_idx, 0].tolist(),
            })

    sample_bank = {
        "tabular_samples": tabular_samples,
        "image_samples": image_samples,
    }
    sample_bank_path = os.path.join(models_dir, "sample_bank.json")
    with open(sample_bank_path, "w") as f:
        json.dump(sample_bank, f, indent=2)
    print(f"  -> Saved sample bank: {sample_bank_path}")

    metadata = {
        "tabular": {
            "model_type": "Multi-Layer Perceptron (Dense 30-32-16-2)",
            "accuracy": float(tab_acc),
            "feature_names": feature_names,
            "target_names": target_names,
        },
        "image": {
            "model_type": "Convolutional Neural Network (Conv-MaxPool-Conv-Dense)",
            "accuracy": float(img_acc),
            "input_shape": [1, 8, 8],
            "num_classes": 10,
        },
        "lrp_rules": ["lrp-0", "lrp-epsilon", "lrp-gamma"],
    }
    metadata_path = os.path.join(models_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  -> Saved metadata: {metadata_path}")

    print("\n[Done] All model artifacts successfully exported.")


if __name__ == "__main__":
    export_all()
