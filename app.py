"""
FastAPI Production Server for Layer-wise Relevance Propagation (LRP)
====================================================================
Exposes high-performance REST API endpoints for real-time model inference
and Layer-wise Relevance Propagation explanations.
"""

import os
import json
import pickle
import time
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from lrp_engine import LRPEngine
from tabular_lrp import build_model as build_tabular_model
from image_lrp import build_convnet as build_image_model


# -----------------------------------------------------------------------------
# Global State
# -----------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
STATE = {
    "start_time": time.time(),
    "tabular_model": None,
    "tabular_engine": None,
    "scaler": None,
    "image_model": None,
    "image_engine": None,
    "metadata": None,
    "sample_bank": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and preprocessors once during startup."""
    print("[Startup] Loading production models from:", MODELS_DIR)
    
    # Metadata
    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            STATE["metadata"] = json.load(f)

    # Sample bank
    sb_path = os.path.join(MODELS_DIR, "sample_bank.json")
    if os.path.exists(sb_path):
        with open(sb_path, "r") as f:
            STATE["sample_bank"] = json.load(f)

    # Scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            STATE["scaler"] = pickle.load(f)

    # Tabular model
    tab_path = os.path.join(MODELS_DIR, "tabular_mlp.pth")
    if os.path.exists(tab_path):
        tab_m = build_tabular_model(input_dim=30)
        tab_m.load_state_dict(torch.load(tab_path, map_location=torch.device("cpu")))
        tab_m.eval()
        STATE["tabular_model"] = tab_m
        STATE["tabular_engine"] = LRPEngine(tab_m)
        print("  -> Tabular MLP loaded.")

    # Image model
    img_path = os.path.join(MODELS_DIR, "image_cnn.pth")
    if os.path.exists(img_path):
        img_m = build_image_model()
        img_m.load_state_dict(torch.load(img_path, map_location=torch.device("cpu")))
        img_m.eval()
        STATE["image_model"] = img_m
        STATE["image_engine"] = LRPEngine(img_m)
        print("  -> Image CNN loaded.")

    yield
    print("[Shutdown] Cleaning up resources.")


app = FastAPI(
    title="Layer-wise Relevance Propagation (LRP) API",
    description="Production REST API providing explainability and feature attribution via LRP for Deep Neural Networks.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for external dashboard or client integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Pydantic Schemas
# -----------------------------------------------------------------------------
class TabularExplainRequest(BaseModel):
    features: List[float] = Field(..., description="List of 30 numerical biometric features")
    is_scaled: bool = Field(False, description="Whether features are already standard-scaled")
    rule: str = Field("lrp-0", description="Propagation rule: 'lrp-0', 'lrp-epsilon', or 'lrp-gamma'")
    target_class: Optional[int] = Field(None, description="Class index to explain (0=Malignant, 1=Benign). If None, explains predicted class.")
    epsilon: float = Field(1e-4, description="Absorber parameter for lrp-epsilon")
    gamma: float = Field(0.25, description="Positive weight multiplier for lrp-gamma")


class FeatureAttribution(BaseModel):
    feature: str
    relevance: float
    raw_value: Optional[float] = None
    is_positive: bool


class TabularExplainResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    probabilities: Dict[str, float]
    target_class: int
    target_label: str
    target_logit: float
    sum_input_relevance: float
    conservation_error: float
    rule: str
    feature_attributions: List[FeatureAttribution]


class ImageExplainRequest(BaseModel):
    pixels_8x8: Optional[List[List[float]]] = Field(None, description="8x8 2D matrix of normalized pixel intensities [0.0 - 1.0]")
    sample_id: Optional[int] = Field(None, description="Optional ID from sample bank (0-9)")
    rule: str = Field("lrp-epsilon", description="Propagation rule: 'lrp-0', 'lrp-epsilon', or 'lrp-gamma'")
    target_class: Optional[int] = Field(None, description="Class index to explain (0-9)")
    competitor_class: Optional[int] = Field(None, description="Optional competitor class for contrastive explanation")
    epsilon: float = Field(1e-3, description="Absorber parameter for lrp-epsilon")
    gamma: float = Field(0.25, description="Positive multiplier for lrp-gamma")


class ImageExplainResponse(BaseModel):
    predicted_class: int
    probabilities: List[float]
    target_class: int
    target_logit: float
    sum_input_relevance: float
    conservation_error: float
    rule: str
    heatmap_8x8: List[List[float]]
    contrastive: Optional[Dict[str, Any]] = None


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/health", summary="Health Check")
def health_check():
    """Returns server status, uptime, and loaded model availability."""
    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - STATE["start_time"], 2),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": {
            "tabular": STATE["tabular_model"] is not None,
            "image": STATE["image_model"] is not None,
        },
    }


@app.get("/api/tabular/metadata", summary="Tabular Model Metadata")
def get_tabular_metadata():
    """Returns feature names, target classes, and model metrics."""
    if not STATE["metadata"]:
        raise HTTPException(status_code=500, detail="Metadata not loaded.")
    return STATE["metadata"]["tabular"]


@app.get("/api/tabular/samples", summary="Tabular Pre-cached Samples")
def get_tabular_samples():
    """Returns curated patient samples for instant testing in UI."""
    if not STATE["sample_bank"]:
        raise HTTPException(status_code=500, detail="Sample bank not loaded.")
    return STATE["sample_bank"]["tabular_samples"]


@app.post("/api/tabular/explain", response_model=TabularExplainResponse, summary="Explain Tabular Prediction")
def explain_tabular(req: TabularExplainRequest):
    """Calculates LRP feature attributions for a 30-dimensional breast cancer sample."""
    if STATE["tabular_engine"] is None or STATE["scaler"] is None:
        raise HTTPException(status_code=503, detail="Tabular LRP engine not initialized.")

    if len(req.features) != 30:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 30 features, received {len(req.features)}.",
        )

    raw_feat = np.array(req.features, dtype=np.float32)
    if req.is_scaled:
        scaled_feat = raw_feat
    else:
        scaled_feat = STATE["scaler"].transform(raw_feat.reshape(1, -1))[0]

    x_tensor = torch.tensor(scaled_feat, dtype=torch.float32).unsqueeze(0)

    # Forward logits and probabilities
    with torch.no_grad():
        logits = STATE["tabular_model"](x_tensor)
        probs = F.softmax(logits, dim=-1).squeeze(0).numpy()

    meta = STATE["metadata"]["tabular"]
    target_names = meta["target_names"]  # ['malignant', 'benign']
    feature_names = meta["feature_names"]

    # LRP explanation
    try:
        exp = STATE["tabular_engine"].explain(
            x_tensor,
            target_class=req.target_class,
            rule=req.rule,
            epsilon=req.epsilon,
            gamma=req.gamma,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"LRP propagation failed: {str(e)}")

    relevance_scores = exp["input_relevance"].squeeze(0).numpy().tolist()
    pred_cls = exp["predicted_class"]
    tgt_cls = exp["target_class"]

    # Format attributions
    attributions = []
    for f_name, r_val, raw_val in zip(feature_names, relevance_scores, req.features):
        attributions.append(
            FeatureAttribution(
                feature=f_name,
                relevance=round(float(r_val), 5),
                raw_value=round(float(raw_val), 4) if not req.is_scaled else None,
                is_positive=bool(r_val >= 0),
            )
        )

    # Sort descending by absolute relevance
    attributions.sort(key=lambda x: abs(x.relevance), reverse=True)

    return TabularExplainResponse(
        predicted_class=pred_cls,
        predicted_label=target_names[pred_cls],
        probabilities={
            target_names[0]: round(float(probs[0]), 4),
            target_names[1]: round(float(probs[1]), 4),
        },
        target_class=tgt_cls,
        target_label=target_names[tgt_cls],
        target_logit=round(float(exp["target_score"]), 4),
        sum_input_relevance=round(float(exp["sum_input_relevance"]), 4),
        conservation_error=float(exp["conservation_error"]),
        rule=exp["rule"],
        feature_attributions=attributions,
    )


@app.get("/api/image/samples", summary="Image Pre-cached Digit Samples")
def get_image_samples():
    """Returns curated 8x8 handwritten digit samples (0-9)."""
    if not STATE["sample_bank"]:
        raise HTTPException(status_code=500, detail="Sample bank not loaded.")
    return STATE["sample_bank"]["image_samples"]


@app.post("/api/image/explain", response_model=ImageExplainResponse, summary="Explain Image Prediction")
def explain_image(req: ImageExplainRequest):
    """Calculates LRP pixel-wise heatmap and optional contrastive attributions for an 8x8 digit."""
    if STATE["image_engine"] is None:
        raise HTTPException(status_code=503, detail="Image LRP engine not initialized.")

    pixels = req.pixels_8x8
    if pixels is None:
        if req.sample_id is not None and STATE["sample_bank"]:
            # Lookup from sample bank
            samples = STATE["sample_bank"]["image_samples"]
            match = next((s for s in samples if s["digit"] == req.sample_id), None)
            if not match:
                match = next((s for s in samples if s["id"] == req.sample_id), None)
            if match:
                pixels = match["pixels_8x8"]
            else:
                pixels = samples[0]["pixels_8x8"]
        else:
            raise HTTPException(status_code=400, detail="Must provide either pixels_8x8 or sample_id.")

    pixels_np = np.array(pixels, dtype=np.float32)
    if pixels_np.shape != (8, 8):
        raise HTTPException(status_code=400, detail=f"Image must be 8x8, got {pixels_np.shape}.")

    x_tensor = torch.tensor(pixels_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Probabilities
    with torch.no_grad():
        logits = STATE["image_model"](x_tensor)
        probs = F.softmax(logits, dim=-1).squeeze(0).numpy().tolist()

    # Primary LRP explanation
    try:
        exp = STATE["image_engine"].explain(
            x_tensor,
            target_class=req.target_class,
            rule=req.rule,
            epsilon=req.epsilon,
            gamma=req.gamma,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"LRP image propagation failed: {str(e)}")

    heatmap = exp["input_relevance"].squeeze().numpy().tolist()

    contrastive_data = None
    if req.competitor_class is not None:
        try:
            exp_comp = STATE["image_engine"].explain(
                x_tensor,
                target_class=req.competitor_class,
                rule=req.rule,
                epsilon=req.epsilon,
                gamma=req.gamma,
            )
            contrastive_data = {
                "competitor_class": req.competitor_class,
                "competitor_logit": round(float(exp_comp["target_score"]), 4),
                "competitor_heatmap_8x8": exp_comp["input_relevance"].squeeze().numpy().tolist(),
                "conservation_error": float(exp_comp["conservation_error"]),
            }
        except Exception:
            pass

    return ImageExplainResponse(
        predicted_class=exp["predicted_class"],
        probabilities=[round(p, 4) for p in probs],
        target_class=exp["target_class"],
        target_logit=round(float(exp["target_score"]), 4),
        sum_input_relevance=round(float(exp["sum_input_relevance"]), 4),
        conservation_error=float(exp["conservation_error"]),
        rule=exp["rule"],
        heatmap_8x8=heatmap,
        contrastive=contrastive_data,
    )


# -----------------------------------------------------------------------------
# Static Files & Dashboard Mount
# -----------------------------------------------------------------------------
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
