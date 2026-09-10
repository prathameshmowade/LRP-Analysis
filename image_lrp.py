"""
Visual / Image Layer-wise Relevance Propagation Analysis
========================================================
Dataset: Handwritten Digits (10 classes: 0 through 9)
Model: Convolutional Neural Network (Conv2d -> ReLU -> MaxPool2d -> Conv2d -> ReLU -> Dense)
Demonstrates:
  1. Pixel-level spatial relevance heatmaps
  2. Conservation property across convolutional and pooling layers
  3. Divergent visualization (Positive relevance = red, Negative relevance = blue)
  4. Contrastive explanation (Target class vs. Counterfactual competitor class)
  5. Rule comparison (LRP-0 vs. LRP-epsilon vs. LRP-gamma)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any

from lrp_engine import LRPEngine


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_digit_data():
    """Load, normalize, and split handwritten digits dataset."""
    digits = load_digits()
    X = digits.images  # shape (1797, 8, 8)
    y = digits.target  # shape (1797,)

    # Normalize pixels to [0, 1]
    X = X / 16.0
    # Add channel dimension: (N, 1, 8, 8)
    X = np.expand_dims(X, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    return train_ds, test_ds, X_test, y_test


def build_convnet() -> nn.Module:
    """Build a Convolutional Neural Network for digit classification."""
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),  # (16, 4, 4)
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 4 * 4, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    return model


def train_convnet(
    model: nn.Module,
    train_ds: TensorDataset,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 0.003,
) -> nn.Module:
    """Train the CNN classifier."""
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def evaluate_convnet(model: nn.Module, test_ds: TensorDataset) -> float:
    """Compute test classification accuracy."""
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            preds = torch.argmax(model(batch_x), dim=-1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / total


def plot_image_lrp_gallery(
    engine: LRPEngine,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sample_indices: List[int],
    save_path: str,
):
    """
    Plots a multi-digit gallery: Input Digit vs. LRP Heatmap vs. Overlay.
    """
    n_samples = len(sample_indices)
    fig, axes = plt.subplots(n_samples, 3, figsize=(9, 2.7 * n_samples))

    if n_samples == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx, s_idx in enumerate(sample_indices):
        img_np = X_test[s_idx, 0]
        true_label = y_test[s_idx]

        x_tensor = torch.tensor(X_test[s_idx : s_idx + 1], dtype=torch.float32)
        exp = engine.explain(x_tensor, rule="lrp-epsilon", epsilon=1e-3)
        rel_map = exp["input_relevance"].squeeze().cpu().numpy()
        pred_label = exp["predicted_class"]

        # Symmetric scale for divergent colormap (center at 0)
        vmax = max(np.max(np.abs(rel_map)), 1e-6)
        vmin = -vmax

        # Column 1: Original Grayscale Image
        axes[row_idx, 0].imshow(img_np, cmap="gray_r", interpolation="nearest")
        axes[row_idx, 0].set_title(f"Sample #{s_idx}\nTrue: {true_label} | Pred: {pred_label}", fontsize=10, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # Column 2: Pure LRP Heatmap
        im = axes[row_idx, 1].imshow(rel_map, cmap="seismic", vmin=vmin, vmax=vmax, interpolation="nearest")
        axes[row_idx, 1].set_title(f"LRP Relevance Map\nLogit: {exp['target_score']:.2f} | $\\Sigma R$: {exp['sum_input_relevance']:.2f}", fontsize=10)
        axes[row_idx, 1].axis("off")
        fig.colorbar(im, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

        # Column 3: Blended Overlay
        axes[row_idx, 2].imshow(img_np, cmap="gray", alpha=0.3, interpolation="nearest")
        im_overlay = axes[row_idx, 2].imshow(rel_map, cmap="seismic", vmin=vmin, vmax=vmax, alpha=0.75, interpolation="nearest")
        axes[row_idx, 2].set_title("Input & Relevance Overlay\n(Red: +, Blue: -)", fontsize=10)
        axes[row_idx, 2].axis("off")

    plt.suptitle("Layer-wise Relevance Propagation (LRP) Visual Explanations", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] Image gallery plot: {save_path}")


def plot_contrastive_explanation(
    engine: LRPEngine,
    x_sample: torch.Tensor,
    true_label: int,
    competitor_class: int,
    save_path: str,
):
    """
    Analyzes why the model predicted the target class instead of a competitor class.
    For instance: Why did it predict '3' instead of '8'?
    """
    exp_pred = engine.explain(x_sample, target_class=true_label, rule="lrp-0")
    exp_comp = engine.explain(x_sample, target_class=competitor_class, rule="lrp-0")

    rel_pred = exp_pred["input_relevance"].squeeze().cpu().numpy()
    rel_comp = exp_comp["input_relevance"].squeeze().cpu().numpy()
    img_np = x_sample.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 1. Original Input
    axes[0].imshow(img_np, cmap="gray_r", interpolation="nearest")
    axes[0].set_title(f"Input Digit (True: {true_label})", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. Relevance for Actual Class
    vmax_1 = max(np.max(np.abs(rel_pred)), 1e-6)
    im1 = axes[1].imshow(rel_pred, cmap="seismic", vmin=-vmax_1, vmax=vmax_1, interpolation="nearest")
    axes[1].set_title(f"Relevance for Class '{true_label}' (Predicted)\nScore: {exp_pred['target_score']:.2f}", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Relevance for Competitor Class
    vmax_2 = max(np.max(np.abs(rel_comp)), 1e-6)
    im2 = axes[2].imshow(rel_comp, cmap="seismic", vmin=-vmax_2, vmax=vmax_2, interpolation="nearest")
    axes[2].set_title(f"Relevance for Class '{competitor_class}' (Competitor)\nScore: {exp_comp['target_score']:.2f}", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle("Contrastive LRP Analysis: Why this class and not another?", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] Contrastive explanation plot: {save_path}")


def plot_image_rule_comparison(
    engine: LRPEngine,
    x_sample: torch.Tensor,
    label: int,
    save_path: str,
):
    """
    Compares LRP-0, LRP-epsilon, and LRP-gamma on the same image sample.
    Demonstrates noise filtering and contrast enhancement across rules.
    """
    rules = [
        ("LRP-0 (Exact Conservation)", "lrp-0", 0.0, 0.0),
        ("LRP-$\\epsilon$ (Noise Suppression)", "lrp-epsilon", 1e-2, 0.0),
        ("LRP-$\\gamma$ (Positive Focus)", "lrp-gamma", 0.0, 0.5),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))

    img_np = x_sample.squeeze().cpu().numpy()
    axes[0].imshow(img_np, cmap="gray_r", interpolation="nearest")
    axes[0].set_title(f"Original Digit ({label})", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    for i, (title, rule, eps, gam) in enumerate(rules):
        exp = engine.explain(x_sample, target_class=label, rule=rule, epsilon=eps, gamma=gam)
        rel = exp["input_relevance"].squeeze().cpu().numpy()
        vmax = max(np.max(np.abs(rel)), 1e-6)

        im = axes[i + 1].imshow(rel, cmap="seismic", vmin=-vmax, vmax=vmax, interpolation="nearest")
        axes[i + 1].set_title(
            f"{title}\nError: {exp['conservation_error']:.1e}",
            fontsize=10.5,
            fontweight="bold",
        )
        axes[i + 1].axis("off")
        fig.colorbar(im, ax=axes[i + 1], fraction=0.046, pad=0.04)

    plt.suptitle("LRP Rule Comparison on 2D Image Attributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] Image rule comparison plot: {save_path}")


def run_image_analysis(output_dir: str = "output_plots"):
    """Full execution pipeline for visual LRP analysis."""
    print("\n" + "=" * 60)
    print("  RUNNING IMAGE LRP ANALYSIS (Handwritten Digits CNN)")
    print("=" * 60)
    set_seed(42)

    # 1. Load data
    train_ds, test_ds, X_test, y_test = load_digit_data()
    print(f"[Data] Training images: {len(train_ds)}, Test images: {len(test_ds)}")

    # 2. Build and train ConvNet
    model = build_convnet()
    print("[Training] Training ConvNet (Conv2d -> MaxPool -> Conv2d -> Dense)...")
    model = train_convnet(model, train_ds, epochs=45)
    acc = evaluate_convnet(model, test_ds)
    print(f"[Evaluation] ConvNet Test Accuracy: {acc * 100:.2f}%")

    # 3. Setup LRP Engine
    engine = LRPEngine(model)

    # 4. Multi-digit gallery
    sample_indices = [3, 10, 15, 25, 42]
    plot_image_lrp_gallery(
        engine,
        X_test,
        y_test,
        sample_indices=sample_indices,
        save_path=os.path.join(output_dir, "image_lrp_gallery.png"),
    )

    # 5. Contrastive explanation (e.g. Digit 3 vs Competitor 8)
    idx_3 = None
    for i in range(len(y_test)):
        if y_test[i] == 3:
            idx_3 = i
            break
    if idx_3 is not None:
        x_3 = torch.tensor(X_test[idx_3 : idx_3 + 1], dtype=torch.float32)
        plot_contrastive_explanation(
            engine,
            x_3,
            true_label=3,
            competitor_class=8,
            save_path=os.path.join(output_dir, "image_contrastive_explanation.png"),
        )

    # 6. Rule comparison on a single digit
    plot_image_rule_comparison(
        engine,
        x_3 if idx_3 is not None else torch.tensor(X_test[0:1], dtype=torch.float32),
        label=3 if idx_3 is not None else int(y_test[0]),
        save_path=os.path.join(output_dir, "image_rule_comparison.png"),
    )

    print(f"[Done] Image LRP analysis successfully finished.\n")


if __name__ == "__main__":
    run_image_analysis()
