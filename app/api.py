# app/api.py - FastAPI Backend
# Production style REST API for digit recognition

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from app.predict import load_model, predict, predict_from_base64

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MNIST Digit Recognizer API",
    description="Production grade handwritten digit recognition",
    version="1.0.0"
)

# CORS middleware
# Allows frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)
templates = Jinja2Templates(directory="templates")

# Add static files to template context
@app.middleware("http")
async def add_static_url(request, call_next):
    response = await call_next(request)
    return response

# Load model once when server starts
logger.info("Loading model...")
model = load_model()
logger.info("Model ready!")


# Request model for base64 input
class Base64Request(BaseModel):
    image: str  # base64 encoded image string


# ── Route 1: Homepage ─────────────────────────────────────────
@app.get("/")
async def index(request: Request):
    """Serves the main web UI"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# ── Route 2: Health Check ─────────────────────────────────────
@app.get("/health")
async def health():
    """
    Health check endpoint.
    Used to verify the service is running.
    Industry standard for all production APIs!
    """
    return {
        "status":  "healthy",
        "model":   "MNIST CNN",
        "version": "1.0.0",
        "accuracy": "99.42%"
    }


# ── Route 3: Predict from Canvas (Base64) ─────────────────────
@app.post("/predict")
async def predict_digit(payload: Base64Request):
    """
    Receives base64 encoded canvas image.
    Returns predicted digit and confidence scores.

    This is the main endpoint used by the web UI canvas!
    """
    try:
        if not payload.image:
            raise HTTPException(
                status_code=400,
                detail="No image provided"
            )

        # Run prediction
        result = predict_from_base64(model, payload.image)

        logger.info(
            f"Prediction: {result['digit']} "
            f"({result['confidence']:.2f}%)"
        )

        return {
            "success":    True,
            "digit":      result["digit"],
            "confidence": result["confidence"],
            "all_probs":  result["all_probs"]
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ── Route 4: Predict from Image File ──────────────────────────
@app.post("/predict-file")
async def predict_from_upload(
    file: UploadFile = File(...)
):
    """
    Receives uploaded image file.
    Returns predicted digit and confidence scores.

    Alternative endpoint for file uploads!
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image!"
            )

        # Read image bytes
        image_bytes = await file.read()
        image       = Image.open(io.BytesIO(image_bytes))

        # Run prediction
        result = predict(model, image)

        logger.info(
            f"File prediction: {result['digit']} "
            f"({result['confidence']:.2f}%)"
        )

        return {
            "success":    True,
            "digit":      result["digit"],
            "confidence": result["confidence"],
            "all_probs":  result["all_probs"],
            "filename":   file.filename
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ── Route 5: Model Info ────────────────────────────────────────
@app.get("/model-info")
async def model_info():
    """Returns information about the loaded model"""
    return {
        "model_name":    "MNISTNet CNN",
        "architecture":  "2 Conv Blocks + Classifier",
        "dataset":       "MNIST 70,000 images",
        "test_accuracy": "99.42%",
        "val_accuracy":  "98.78%",
        "input_size":    "28x28 grayscale",
        "output":        "10 classes (0-9)",
        "framework":     "PyTorch"
    }


# Run the app
if __name__ == "__main__":
    import uvicorn
    import yaml

    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    uvicorn.run(
        "app.api:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=config["api"]["debug"]
    )