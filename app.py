"""
FastAPI Production Server for Layer-wise Relevance Propagation (LRP)
====================================================================
High-performance REST API and Serverless Web Application for LRP explainability.
Powered by the ultra-lightweight NumPyLRPEngine (<35 MB footprint for Vercel/Cloud deployment)
with automatic PyTorch fallback if available.
"""

import os
import json
import time
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from lrp_numpy import NumPyLRPEngine

# -----------------------------------------------------------------------------
# Global State & Engine Initialization
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Lightweight engine (<35 MB, boots in ~0.05s)
ENGINE = NumPyLRPEngine(models_dir=MODELS_DIR)
START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Initializing LRP Explainability Engine...")
    print(f"  -> Models dir: {MODELS_DIR}")
    print(f"  -> Metadata loaded: {ENGINE.metadata is not None}")
    print(f"  -> Weights loaded: {ENGINE.weights is not None}")
    yield
    print("[Shutdown] Cleaned up resources.")


app = FastAPI(
    title="Layer-wise Relevance Propagation (LRP) API",
    description="Production REST API providing explainability and feature attribution via LRP for Deep Neural Networks.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin frontend clients
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
    target_class: Optional[int] = Field(None, description="Class index to explain (0=Malignant, 1=Benign)")
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
    sample_id: Optional[int] = Field(None, description="Optional digit or ID from sample bank (0-9)")
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
# REST API Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/health", summary="Health Check")
def health_check():
    """Returns server status, uptime, and loaded model availability."""
    return {
        "status": "healthy",
        "runtime": "lightweight-serverless",
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "models_loaded": {
            "tabular": ENGINE.weights is not None and "tab_0.weight" in ENGINE.weights,
            "image": ENGINE.weights is not None and "img_0.weight" in ENGINE.weights,
        },
    }


@app.get("/api/tabular/metadata", summary="Tabular Model Metadata")
def get_tabular_metadata():
    """Returns feature names, target classes, and model metrics."""
    if not ENGINE.metadata:
        raise HTTPException(status_code=500, detail="Metadata not loaded.")
    return ENGINE.metadata["tabular"]


@app.get("/api/tabular/samples", summary="Tabular Pre-cached Samples")
def get_tabular_samples():
    """Returns curated patient samples for instant testing in UI."""
    if not ENGINE.sample_bank:
        raise HTTPException(status_code=500, detail="Sample bank not loaded.")
    return ENGINE.sample_bank["tabular_samples"]


@app.post("/api/tabular/explain", response_model=TabularExplainResponse, summary="Explain Tabular Prediction")
def explain_tabular(req: TabularExplainRequest):
    """Calculates LRP feature attributions for a 30-dimensional breast cancer sample."""
    if len(req.features) != 30:
        raise HTTPException(status_code=400, detail=f"Expected 30 features, received {len(req.features)}.")

    try:
        res = ENGINE.explain_tabular(
            features=req.features,
            is_scaled=req.is_scaled,
            rule=req.rule,
            target_class=req.target_class,
            epsilon=req.epsilon,
            gamma=req.gamma,
        )
        return TabularExplainResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"LRP propagation failed: {str(e)}")


@app.get("/api/image/samples", summary="Image Pre-cached Digit Samples")
def get_image_samples():
    """Returns curated 8x8 handwritten digit samples (0-9)."""
    if not ENGINE.sample_bank:
        raise HTTPException(status_code=500, detail="Sample bank not loaded.")
    return ENGINE.sample_bank["image_samples"]


@app.post("/api/image/explain", response_model=ImageExplainResponse, summary="Explain Image Prediction")
def explain_image(req: ImageExplainRequest):
    """Calculates LRP pixel-wise heatmap and optional contrastive attributions for an 8x8 digit."""
    pixels = req.pixels_8x8
    if pixels is None:
        if req.sample_id is not None and ENGINE.sample_bank:
            samples = ENGINE.sample_bank["image_samples"]
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

    try:
        res = ENGINE.explain_image(
            pixels_8x8=pixels,
            rule=req.rule,
            target_class=req.target_class,
            competitor_class=req.competitor_class,
            epsilon=req.epsilon,
            gamma=req.gamma,
        )
        return ImageExplainResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"LRP image propagation failed: {str(e)}")


# -----------------------------------------------------------------------------
# Root Dashboard & Static Files
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, summary="Web Dashboard Root")
def serve_dashboard():
    """Serves the interactive LRP Explainability Studio HTML page."""
    html_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>LRP Explainability Studio</h1>"


if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
