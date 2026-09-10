"""
Tabular Layer-wise Relevance Propagation Analysis
=================================================
Dataset: Breast Cancer Wisconsin Diagnostic Dataset (30 features)
Task: Binary classification (Malignant vs. Benign)
Demonstrates:
  1. Training a PyTorch MLP classifier
  2. Calculating local feature attributions using LRP-0, LRP-epsilon, LRP-gamma
  3. Strict conservation property verification
  4. Visualizing positive vs. negative feature contributions for individual cases
  5. Aggregating global feature importance over test cohort
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any

from lrp_engine import LRPEngine


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_and_preprocess_data():
    """Load Breast Cancer dataset, split, scale, and convert to PyTorch tensors."""
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names  # ['malignant', 'benign']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    return (
        train_dataset,
        test_dataset,
        X_test_scaled,
        y_test,
        feature_names,
        target_names,
        scaler,
    )


def build_model(input_dim: int = 30) -> nn.Module:
    """Build a Multi-Layer Perceptron (MLP) classifier."""
    model = nn.Sequential(
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )
    return model


def train_model(
    model: nn.Module,
    train_dataset: TensorDataset,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 0.005,
) -> nn.Module:
    """Train the MLP model using CrossEntropyLoss and Adam optimizer."""
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def evaluate_model(model: nn.Module, test_dataset: TensorDataset) -> float:
    """Evaluate accuracy on the test set."""
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            preds = torch.argmax(model(batch_x), dim=-1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    acc = correct / total
    return acc


def plot_local_explanation(
    exp: Dict[str, Any],
    feature_names: np.ndarray,
    target_names: np.ndarray,
    sample_idx: int,
    true_label: int,
    save_path: str,
    top_k: int = 12,
):
    """
    Plot local feature contributions: red bars for positive (evidence towards target class),
    blue bars for negative (evidence opposing target class).
    """
    relevances = exp["input_relevance"].squeeze(0).cpu().numpy()
    pred_class = exp["predicted_class"]
    target_class = exp["target_class"]
    pred_name = target_names[pred_class]
    true_name = target_names[true_label]

    # Sort by absolute magnitude to highlight most influential features
    abs_indices = np.argsort(np.abs(relevances))[::-1][:top_k]
    top_features = [feature_names[i] for i in abs_indices][::-1]
    top_rel = relevances[abs_indices][::-1]

    colors = ["#2b6cb0" if r < 0 else "#c53030" for r in top_rel]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6.5))

    y_pos = np.arange(len(top_features))
    bars = ax.barh(y_pos, top_rel, color=colors, edgecolor="black", alpha=0.85, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features, fontsize=10, fontweight="medium")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)

    # Annotate values on bars
    for bar in bars:
        width = bar.get_width()
        x_loc = width + (0.01 if width >= 0 else -0.01)
        ha = "left" if width >= 0 else "right"
        ax.annotate(
            f"{width:+.3f}",
            (x_loc, bar.get_y() + bar.get_height() / 2),
            va="center",
            ha=ha,
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_xlabel("Relevance Score $R_i$ (Contribution toward prediction)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"LRP Local Feature Attribution (Sample #{sample_idx})\n"
        f"True Class: {true_name.upper()} | Predicted Class: {pred_name.upper()} (Logit: {exp['target_score']:.2f})\n"
        f"Rule: {exp['rule']} | Conservation Error: {exp['conservation_error']:.2e}",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#c53030", edgecolor="black", label=f"Supports '{pred_name}' (Positive)"),
        Patch(facecolor="#2b6cb0", edgecolor="black", label=f"Opposes '{pred_name}' (Negative)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] Local explanation plot: {save_path}")


def plot_global_importance(
    all_relevances: np.ndarray,
    feature_names: np.ndarray,
    save_path: str,
    top_k: int = 15,
):
    """
    Plot global feature importance by computing mean absolute relevance over all test instances.
    """
    mean_abs_rel = np.mean(np.abs(all_relevances), axis=0)
    sorted_idx = np.argsort(mean_abs_rel)[::-1][:top_k]

    top_names = [feature_names[i] for i in sorted_idx][::-1]
    top_scores = mean_abs_rel[sorted_idx][::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(top_names))
    bars = ax.barh(y_pos, top_scores, color="#2c7a7b", edgecolor="black", alpha=0.85, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=10, fontweight="medium")

    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f"{width:.3f}",
            (width + 0.005, bar.get_y() + bar.get_height() / 2),
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_xlabel("Mean Absolute Relevance $\\mathbb{E}[|R_i|]$", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Global Feature Importance via Layer-wise Relevance Propagation\n(Averaged across Test Cohort)",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] Global feature importance plot: {save_path}")


def plot_rule_comparison(
    lrp_engine: LRPEngine,
    x_tensor: torch.Tensor,
    feature_names: np.ndarray,
    save_path: str,
    top_k: int = 10,
):
    """
    Compare LRP-0, LRP-epsilon, and LRP-gamma on the same test sample.
    """
    exp_0 = lrp_engine.explain(x_tensor, rule="lrp-0")
    exp_eps = lrp_engine.explain(x_tensor, rule="lrp-epsilon", epsilon=1e-1)
    exp_gam = lrp_engine.explain(x_tensor, rule="lrp-gamma", gamma=0.5)

    r0 = exp_0["input_relevance"].squeeze(0).cpu().numpy()
    reps = exp_eps["input_relevance"].squeeze(0).cpu().numpy()
    rgam = exp_gam["input_relevance"].squeeze(0).cpu().numpy()

    # Pick top features by LRP-0 magnitude
    top_idx = np.argsort(np.abs(r0))[::-1][:top_k]
    names = [feature_names[i] for i in top_idx]

    y = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.barh(y + width, [r0[i] for i in top_idx], width, label=f"LRP-0 (Error: {exp_0['conservation_error']:.1e})", color="#3182ce")
    ax.barh(y, [reps[i] for i in top_idx], width, label=f"LRP-$\\epsilon$ ($\\epsilon=0.1$)", color="#38a169")
    ax.barh(y - width, [rgam[i] for i in top_idx], width, label=f"LRP-$\\gamma$ ($\\gamma=0.5$)", color="#dd6b20")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Relevance Score", fontsize=11, fontweight="bold")
    ax.set_title("Comparison of LRP Propagation Rules on the Same Sample", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] LRP rule comparison plot: {save_path}")


def run_tabular_analysis(output_dir: str = "output_plots"):
    """Full execution pipeline for tabular LRP analysis."""
    print("\n" + "=" * 60)
    print("  RUNNING TABULAR LRP ANALYSIS (Breast Cancer Dataset)")
    print("=" * 60)
    set_seed(42)

    # 1. Prepare data
    (
        train_data,
        test_data,
        X_test,
        y_test,
        feature_names,
        target_names,
        scaler,
    ) = load_and_preprocess_data()
    print(f"[Data] Training samples: {len(train_data)}, Test samples: {len(test_data)}, Features: {len(feature_names)}")

    # 2. Build and train model
    model = build_model(input_dim=len(feature_names))
    print("[Training] Training PyTorch MLP classifier...")
    model = train_model(model, train_data, epochs=70)
    test_acc = evaluate_model(model, test_data)
    print(f"[Evaluation] Model Test Accuracy: {test_acc * 100:.2f}%")

    # 3. Setup LRP Engine
    engine = LRPEngine(model)

    # 4. Local explanation for a Malignant sample (class 0) and Benign sample (class 1)
    malignant_indices = np.where(y_test == 0)[0]
    benign_indices = np.where(y_test == 1)[0]

    idx_mal = malignant_indices[0]
    idx_ben = benign_indices[0]

    x_mal = torch.tensor(X_test[idx_mal], dtype=torch.float32)
    x_ben = torch.tensor(X_test[idx_ben], dtype=torch.float32)

    exp_mal = engine.explain(x_mal, rule="lrp-0")
    exp_ben = engine.explain(x_ben, rule="lrp-0")

    print(f"\n[Sample #{idx_mal} - Malignant Case]")
    print(f"  Target Logit: {exp_mal['target_score']:.4f}")
    print(f"  Input Relevance Sum: {exp_mal['sum_input_relevance']:.4f}")
    print(f"  Conservation Error: {exp_mal['conservation_error']:.2e}")

    print(f"\n[Sample #{idx_ben} - Benign Case]")
    print(f"  Target Logit: {exp_ben['target_score']:.4f}")
    print(f"  Input Relevance Sum: {exp_ben['sum_input_relevance']:.4f}")
    print(f"  Conservation Error: {exp_ben['conservation_error']:.2e}")

    # 5. Plot local explanations
    plot_local_explanation(
        exp_mal,
        feature_names,
        target_names,
        sample_idx=idx_mal,
        true_label=y_test[idx_mal],
        save_path=os.path.join(output_dir, "tabular_local_malignant.png"),
    )
    plot_local_explanation(
        exp_ben,
        feature_names,
        target_names,
        sample_idx=idx_ben,
        true_label=y_test[idx_ben],
        save_path=os.path.join(output_dir, "tabular_local_benign.png"),
    )

    # 6. Global importance across all test samples
    print("\n[Global Analysis] Computing relevance across full test set...")
    all_relevances = []
    for i in range(len(X_test)):
        xi = torch.tensor(X_test[i], dtype=torch.float32)
        exp_i = engine.explain(xi, rule="lrp-0")
        all_relevances.append(exp_i["input_relevance"].squeeze(0).cpu().numpy())
    all_relevances = np.array(all_relevances)

    plot_global_importance(
        all_relevances,
        feature_names,
        save_path=os.path.join(output_dir, "tabular_global_importance.png"),
    )

    # 7. Comparison of rules (LRP-0 vs LRP-eps vs LRP-gamma)
    plot_rule_comparison(
        engine,
        x_mal,
        feature_names,
        save_path=os.path.join(output_dir, "tabular_rule_comparison.png"),
    )

    print(f"[Done] Tabular LRP analysis successfully finished.\n")


if __name__ == "__main__":
    run_tabular_analysis()
