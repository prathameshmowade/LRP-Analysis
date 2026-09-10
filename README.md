# Layer-wise Relevance Propagation (LRP) Analysis

**Course**: Explainable Artificial Intelligence (XAI) — SEM 7  
**Assignment**: Term Assignment Evaluation 1 (TAE 1)  
**Topic**: Apply Layer-wise Relevance Propagation to a trained model and analyze feature contributions toward predictions.

---

## 1. Executive Summary

This repository presents a mathematically rigorous, end-to-end implementation of **Layer-wise Relevance Propagation (LRP)** applied to trained deep neural networks. The analysis encompasses two complementary modalities:

1. **Tabular Feature Attribution (MLP Classifier)**: Applied to the real-world **Breast Cancer Wisconsin Diagnostic dataset** (30 morphological features) to quantify positive and negative biomarker contributions toward malignant and benign clinical diagnoses.
2. **Visual Feature Attribution (Convolutional Neural Network)**: Applied to **2D Handwritten Digits** to inspect spatial pixel-level relevance heatmaps, perform contrastive counterfactual explanations (e.g., *Why digit 3 instead of 8?*), and demonstrate noise-suppression rules.

All algorithms are implemented from first principles in **PyTorch** and validated against the foundational **Layer-wise Conservation Principle** ($\sum_i R_i^{(0)} \approx f(x)_c$).

---

## 2. Mathematical Background & Propagation Rules

Layer-wise Relevance Propagation (Bach et al., 2015; Montavon et al., 2019) explains a neural network's scalar prediction $f(x)_c$ by decomposing it into relevance scores $R_j^{(l)}$ for all neurons $j$ across layer $l$, propagating backward from layer $L$ to input features $0$.

### The Conservation Property
Relevance is conserved across adjacent layers:
$$\sum_{i} R_i^{(0)} = \dots = \sum_{j} R_j^{(l)} = \sum_{k} R_k^{(l+1)} = \dots = R_c^{(L)} = f(x)_c$$

### 1. LRP-$0$ (Standard Conservation Rule)
Distributes relevance strictly proportional to the contribution of neuron $j$ to the activation of neuron $k$:
$$z_{jk} = a_j w_{jk}, \quad z_k = \sum_j z_{jk}$$
$$R_j^{(l)} = \sum_k \frac{a_j w_{jk}}{z_k} R_k^{(l+1)}$$
- **Properties**: Strictly preserves the conservation law ($\text{Error} < 10^{-6}$).
- **Usage**: Baseline attribution for verifying theoretical conservation.

### 2. LRP-$\epsilon$ (Noise Absorption & Numerical Stability)
Adds a stabilizer/absorber $\epsilon > 0$ to the denominator:
$$R_j^{(l)} = \sum_k \frac{a_j w_{jk}}{z_k + \epsilon \cdot \text{sign}(z_k)} R_k^{(l+1)}$$
- **Properties**: Suppresses weak, ambiguous, or noisy activations. When $\epsilon \to \infty$, relevance flows primarily through dominant, high-magnitude pathways.
- **Usage**: Produces cleaner image heatmaps and filters background noise.

### 3. LRP-$\gamma$ (Positive Evidence Enhancement)
Upweights positive weights by factor $\gamma \ge 0$:
$$w_{jk}^+ = w_{jk} + \gamma \max(0, w_{jk})$$
$$R_j^{(l)} = \sum_k \frac{a_j w_{jk}^+}{\sum_{j'} a_{j'} w_{j'k}^+} R_k^{(l+1)}$$
- **Properties**: Places greater weight on positive evidence supporting the decision, making attributions sharper and more human-interpretable.

---

## 3. Architecture & Project Layout

```
SEM 7/XAI/TAE 1/
│
├── lrp_engine.py                 # Core PyTorch LRP engine (Linear, Conv2d, ReLU, MaxPool, Flatten)
├── tabular_lrp.py                # Tabular pipeline: Breast Cancer MLP, local & global attributions
├── image_lrp.py                  # Image pipeline: Handwritten Digits CNN, spatial heatmaps
├── run_all.py                    # Unified CLI runner executing both pipelines
├── LRP_Analysis_Notebook.ipynb   # Interactive self-contained Jupyter Notebook
├── README.md                     # Academic project documentation & report
│
└── output_plots/                 # High-resolution generated visualizations (300 DPI)
    ├── tabular_local_malignant.png
    ├── tabular_local_benign.png
    ├── tabular_global_importance.png
    ├── tabular_rule_comparison.png
    ├── image_lrp_gallery.png
    ├── image_contrastive_explanation.png
    └── image_rule_comparison.png
```

---

## 4. Experimental Results & Feature Analysis

### A. Tabular Feature Attribution (Breast Cancer MLP)
- **Architecture**: `Dense(30 -> 32) -> ReLU -> Dense(32 -> 16) -> ReLU -> Dense(16 -> 2)`
- **Evaluation Accuracy**: **95.61%** on held-out test cohort.
- **Conservation Check**:
  - Sample #0 (Malignant): Target Logit = $40.6220$, $\sum R_i = 40.6220$, Error = $3.81 \times 10^{-6}$.
  - Sample #1 (Benign): Target Logit = $14.0646$, $\sum R_i = 14.0646$, Error = $0.00 \times 10^{0}$.

#### Key Feature Contributions:
- **Malignant Diagnosis Drivers**: The features providing strongest positive relevance ($R_i > 0$) toward malignancy are `worst concave points`, `worst perimeter`, `mean concave points`, and `worst radius`. Larger cell dimensions and severe structural irregularities strongly drive the network toward predicting malignancy.
- **Benign Diagnosis Drivers**: Normal boundary smoothness (`worst smoothness`), smaller cell radii, and regular cellular geometry act as strong negative counter-evidence against malignancy (or positive evidence toward benign).
- **Global Importance**: Across the 114 test patients, `worst perimeter`, `worst concave points`, and `worst area` exhibited the highest mean absolute relevance $\mathbb{E}[|R_i|]$, confirming clinical diagnostic intuition.

### B. Visual Heatmap Attribution (Digit Recognition CNN)
- **Architecture**: `Conv2d(1->16, 3x3) -> ReLU -> MaxPool2d(2x2) -> Conv2d(16->32, 3x3) -> ReLU -> Flatten -> Dense(512->64) -> ReLU -> Dense(64->10)`
- **Evaluation Accuracy**: **97.22%** on test digits.

#### Key Spatial Findings:
- **Stroke Localization**: For digit '3', positive relevance (red) is intensely concentrated along the top horizontal bar, central intersection cusp, and lower curved basin.
- **Contrastive Analysis (Digit 3 vs. Competitor 8)**:
  - Explaining the correct class '3' yields positive relevance throughout the active strokes.
  - Explaining competitor class '8' reveals strong **negative relevance (blue)** in the left vertical margin where strokes would be required to close the loops for an '8'. The network actively recognized the *absence* of left closures as evidence against class '8'.
- **Rule Comparison**: LRP-0 produces the raw conserved signal; LRP-$\epsilon$ eliminates peripheral background scatter; LRP-$\gamma$ sharpens the central stroke contours.

---

## 5. How to Run

### 1. Run Complete Pipeline via CLI
Execute the automated suite in PowerShell / Terminal:
```powershell
python run_all.py
```
This runs both tabular and visual analyses, validates conservation, and populates `output_plots/`.

### 2. Run Tabular Analysis Standalone
```powershell
python tabular_lrp.py
```

### 3. Run Image Analysis Standalone
```powershell
python image_lrp.py
```

### 4. Interactive Jupyter Notebook
Open and run all cells in `LRP_Analysis_Notebook.ipynb` using JupyterLab, VS Code, or Google Colab:
```powershell
jupyter notebook LRP_Analysis_Notebook.ipynb
```

### 5. Production Server & Interactive Dashboard
Start the production FastAPI server:
```powershell
python start_server.py
```
- **Web Dashboard**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 6. Automated Integration Tests
Verify REST endpoints and mathematical conservation laws:
```powershell
python test_deployment.py
```

### 7. Docker Deployment
```powershell
# Build and run standalone container
docker build -t lrp-studio:latest .
docker run -p 8000:8000 lrp-studio:latest

# Or launch full stack (FastAPI + Streamlit) via Docker Compose
docker-compose up --build -d
```

---

## 6. References
1. Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K. R., & Samek, W. (2015). *On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation*. **PLOS ONE**, 10(7), e0130140.
2. Montavon, G., Binder, A., Lapuschkin, S., Samek, W., & Müller, K. R. (2019). *Layer-Wise Relevance Propagation: An Overview*. In **Explainable AI: Interpreting, Explaining and Visualizing Deep Learning** (pp. 193-209). Springer, Cham.
3. Samek, W., Montavon, G., Lapuschkin, S., Anders, C. J., & Müller, K. R. (2021). *Explaining Deep Neural Networks and Beyond: A Review of Methods and Applications*. **Proceedings of the IEEE**, 109(3), 247-278.
