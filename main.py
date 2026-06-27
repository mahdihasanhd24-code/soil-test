import os
import json
import threading
import uuid
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image
import joblib
import tempfile

# Import local modules
from split_dataset import split_data
from train_model import train_and_evaluate
from feature_extractor import extract_all_features_dict, get_feature_vector_by_setting

app = FastAPI(title="TerraCNN - Soil Image Classification API")

# CORS middleware — required when frontend (Vercel) and backend (Render) are on different domains
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:5500",   # VS Code Live Server
        "http://localhost:5500",
        # Add your Vercel URL here after deployment, e.g.:
        # "https://soil-test.vercel.app",
        "*",  # Temporarily allow all origins — replace with specific Vercel URL in production
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "DataSet", "Soil Dataset")
SPLIT_DIR = os.path.join(BASE_DIR, "dataset_split")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATUS_FILE = os.path.join(MODELS_DIR, "training_status.json")
METRICS_FILE = os.path.join(MODELS_DIR, "soil_metrics.json")
MODEL_FILE = os.path.join(MODELS_DIR, "soil_model.joblib")

# Global variables to cache the loaded model and classes
cached_model = None
cached_classes = []
cached_model_mtime = 0

# Descriptions for soil classes to make predictions rich and professional
SOIL_DESCRIPTIONS = {
    "Black Soil": {
        "description": "Also known as Chernozem or Black Cotton soil. It is highly argillaceous (clay-rich), dark in color, and possesses a high moisture-retention capacity.",
        "properties": "Rich in calcium, carbonate, potash, and magnesium; low in phosphorus and nitrogen. Swells when wet and cracks when dry, aiding self-aeration.",
        "suitability": "Highly suitable for cotton, wheat, linseed, millets, tobacco, and oilseeds."
    },
    "Cinder Soil": {
        "description": "A vesicular, dark volcanic soil consisting primarily of cinders and ash fragments. It is highly porous and lightweight.",
        "properties": "Excellent drainage, low organic matter, highly aerated. Does not retain nutrients or water well on its own.",
        "suitability": "Ideal for cacti, succulents, bonsai, or as a soil-conditioning additive to improve soil drainage and prevent compaction."
    },
    "Laterite Soil": {
        "description": "Formed in tropical regions with high temperatures and heavy seasonal rainfall, causing intense leaching. It is rich in iron and aluminum oxides, giving it a rusty-red appearance.",
        "properties": "Highly acidic, low clay content, poor water-holding capacity, and very low organic matter/humus content.",
        "suitability": "Well-suited for plantation crops such as tea, coffee, cashew nuts, rubber, and coconuts with appropriate fertilizers."
    },
    "Loam Soil": {
        "description": "Often considered the ideal agricultural soil. It contains a balanced proportion of sand (approx. 40%), silt (approx. 40%), and clay (approx. 20%).",
        "properties": "Excellent nutrient content, friable structure, retains moisture well while allowing excess water to drain freely. Easy to till.",
        "suitability": "Outstanding for almost all vegetables, fruits, ornamental plants, grains, and general gardening."
    },
    "Peat Soil": {
        "description": "A dark, highly organic soil composed of partially decomposed organic matter accumulated in wet, acidic, anaerobic conditions (peat bogs).",
        "properties": "Extremely high moisture retention, low pH (highly acidic), high organic content, and poor structural drainage.",
        "suitability": "Excellent for acid-loving plants such as blueberries, cranberries, heathers, and rhododendrons. Used to condition soil organic levels."
    },
    "Yellow Soil": {
        "description": "A clayey or sandy loam soil that derives its yellow color from the presence of hydrated iron oxides (limonite). Typically found in humid subtropical climates.",
        "properties": "Moderate to low fertility, acidic to neutral, decent drainage but prone to erosion. Poor in organic matter and nitrogen.",
        "suitability": "Supports rice cultivation, citrus fruits, tea, and sweet potatoes with proper organic fertilization."
    }
}

class TrainRequest(BaseModel):
    classes: List[str] = Field(..., min_length=2, description="Classes to train on (minimum 2)")
    epochs: int = Field(10, ge=1, le=100, description="Number of training epochs")
    batch_size: int = Field(32, ge=8, le=128, description="Batch size")
    learning_rate: float = Field(0.001, ge=0.00001, le=0.1, description="Learning rate for Adam optimizer")

# Helper function to check model cache and reload if needed
def get_model():
    global cached_model, cached_classes, cached_model_mtime
    
    if not os.path.exists(MODEL_FILE) or not os.path.exists(METRICS_FILE):
        return None, []
        
    mtime = os.path.getmtime(MODEL_FILE)
    if cached_model is None or mtime > cached_model_mtime:
        try:
            print("Loading joblib model pipeline from disk...")
            pipeline = joblib.load(MODEL_FILE)
            cached_model = pipeline # holds the scaler and the classifier model
            cached_classes = pipeline["classes"]
            cached_model_mtime = mtime
            print(f"Model loaded successfully with classes: {cached_classes}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, []
            
    return cached_model, cached_classes

@app.get("/api/classes")
def get_available_classes():
    if not os.path.exists(DATASET_DIR):
        raise HTTPException(status_code=404, detail="Soil Dataset directory not found.")
    classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    return {"classes": sorted(classes)}

@app.post("/api/split")
def trigger_split():
    try:
        counts = split_data(DATASET_DIR, SPLIT_DIR)
        return {"status": "success", "counts": counts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Split failed: {str(e)}")

@app.get("/api/split-status")
def get_split_status():
    if not os.path.exists(SPLIT_DIR):
        return {"split_done": False, "counts": {}}
        
    splits = ["train", "val", "test"]
    counts = {}
    
    try:
        # Check folders inside splits
        for split in splits:
            split_path = os.path.join(SPLIT_DIR, split)
            if not os.path.exists(split_path):
                return {"split_done": False, "counts": {}}
                
            classes = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
            for cls in classes:
                if cls not in counts:
                    counts[cls] = {"train": 0, "val": 0, "test": 0}
                cls_path = os.path.join(split_path, cls)
                counts[cls][split] = len(os.listdir(cls_path))
                
        return {"split_done": True, "counts": counts}
    except Exception as e:
        return {"split_done": False, "error": str(e)}

@app.post("/api/upload-dataset")
async def upload_dataset_images(class_name: str = Form(...), files: List[UploadFile] = File(...)):
    if not class_name or not class_name.strip():
        raise HTTPException(status_code=400, detail="Class name cannot be empty.")
        
    class_dir = os.path.join(DATASET_DIR, class_name.strip())
    os.makedirs(class_dir, exist_ok=True)
    
    saved_count = 0
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in valid_extensions:
            continue
            
        # Use sanitized original filename to prevent duplicate UUID accumulation
        safe_name = os.path.basename(file.filename)
        if not safe_name:
            safe_name = f"upload_{uuid.uuid4().hex}{ext}"
            
        file_path = os.path.join(class_dir, safe_name)
        
        try:
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            saved_count += 1
        except Exception as e:
            print(f"Error saving file {file.filename}: {e}")
            
    return {"status": "success", "class_name": class_name, "uploaded_count": saved_count}

def run_async_train(classes: List[str], epochs: int, batch_size: int, learning_rate: float):
    try:
        train_and_evaluate(
            selected_classes=classes,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            split_dir=SPLIT_DIR,
            models_dir=MODELS_DIR
        )
    except Exception as e:
        print(f"Async training error: {e}")

@app.post("/api/train")
def train_model(req: TrainRequest, background_tasks: BackgroundTasks):
    # Always re-split dataset to ensure new uploads are partitioned
    try:
        print("Re-splitting dataset to include new uploads...")
        split_data(DATASET_DIR, SPLIT_DIR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to split dataset: {str(e)}")
            
    # Check if already training
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r') as f:
                status_data = json.load(f)
                if status_data.get("status") == "training":
                    raise HTTPException(status_code=400, detail="Training is already in progress.")
        except:
            pass
            
    # Write initial state
    os.makedirs(MODELS_DIR, exist_ok=True)
    initial_status = {
        "status": "training",
        "progress": 0,
        "current_epoch": 0,
        "total_epochs": req.epochs,
        "metrics": {"loss": 0.0, "accuracy": 0.0, "val_loss": 0.0, "val_accuracy": 0.0},
        "history": []
    }
    with open(STATUS_FILE, 'w') as f:
        json.dump(initial_status, f, indent=2)
        
    # Start training in background
    background_tasks.add_task(run_async_train, req.classes, req.epochs, req.batch_size, req.learning_rate)
    return {"status": "started", "message": "Training started in background."}

@app.get("/api/train-status")
def get_training_status():
    if not os.path.exists(STATUS_FILE):
        return {"status": "idle"}
    try:
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/metrics")
def get_metrics():
    if not os.path.exists(METRICS_FILE):
        raise HTTPException(status_code=404, detail="No trained model metrics found. Please train the model first.")
    try:
        with open(METRICS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read metrics: {str(e)}")

@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
    pipeline, classes = get_model()
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Model is not trained or loaded. Please train a model first.")
        
    try:
        # Read uploaded image content
        image_content = await file.read()
        
        # Write to temporary file for feature extraction
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file.write(image_content)
            temp_path = temp_file.name
            
        try:
            # Preprocess and extract features
            feats_dict = extract_all_features_dict(temp_path)
            feat_vector = get_feature_vector_by_setting(feats_dict, setting_num=12)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        # Scale and predict
        scaler = pipeline["scaler"]
        classifier = pipeline["model"]
        
        feat_vector_scaled = scaler.transform([feat_vector])
        
        # Get probabilities and predicted class
        probs = classifier.predict_proba(feat_vector_scaled)[0]
        max_idx = int(np.argmax(probs))
        
        predicted_class = classes[max_idx]
        confidence = float(probs[max_idx]) * 100
        
        # Get matching probabilities for all classes
        probabilities = {classes[i]: float(probs[i]) * 100 for i in range(len(classes))}
        
        # Get soil description
        soil_info = SOIL_DESCRIPTIONS.get(predicted_class, {
            "description": "Classified soil type details.",
            "properties": "N/A",
            "suitability": "N/A"
        })
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "info": soil_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Add io import for file processing
import io

# Serve static files for frontend UI
static_path = os.path.join(BASE_DIR, "static")
os.makedirs(static_path, exist_ok=True)
app.mount("/", StaticFiles(directory=static_path, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Read port from environment variable (required by Render.com)
    # Falls back to 8080 for local development
    port = int(os.environ.get("PORT", 8080))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    uvicorn.run("main:app", host=host, port=port, reload=False)
