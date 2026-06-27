import os
import json
import threading
import uuid
import io
import pickle
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image

# Import local modules
from split_dataset import split_data
from train_model import train_and_evaluate, extract_features

app = FastAPI(title="TerraCNN - Soil Image Classification API")

# CORS middleware — required when frontend (Vercel) and backend (Render) are on different domains
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "DataSet")
SPLIT_DIR = os.path.join(BASE_DIR, "dataset_split")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATUS_FILE = os.path.join(MODELS_DIR, "training_status.json")
METRICS_FILE = os.path.join(MODELS_DIR, "soil_metrics.json")

# Pickled model paths
SVM_MODEL_FILE = os.path.join(MODELS_DIR, "svm_model.pkl")
KNN_MODEL_FILE = os.path.join(MODELS_DIR, "knn_model.pkl")
DT_MODEL_FILE = os.path.join(MODELS_DIR, "dt_model.pkl")

# Global variables to cache the loaded models and classes
cached_models = {
    "svm": None,
    "knn": None,
    "dt": None
}
cached_classes = []
cached_models_mtime = 0

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
def get_model(model_type="svm"):
    global cached_models, cached_classes, cached_models_mtime
    
    model_files = {
        "svm": SVM_MODEL_FILE,
        "knn": KNN_MODEL_FILE,
        "dt": DT_MODEL_FILE
    }
    
    target_file = model_files.get(model_type)
    if not target_file or not os.path.exists(target_file) or not os.path.exists(METRICS_FILE):
        return None, []
        
    mtime = os.path.getmtime(target_file)
    if cached_models[model_type] is None or mtime > cached_models_mtime:
        try:
            print(f"Loading pickled {model_type} model from disk...")
            with open(target_file, 'rb') as f:
                cached_models[model_type] = pickle.load(f)
            with open(METRICS_FILE, 'r') as f:
                metrics_data = json.load(f)
                cached_classes = metrics_data["confusion_matrix"]["classes"]
            cached_models_mtime = mtime
            print(f"Model {model_type} loaded successfully with classes: {cached_classes}")
        except Exception as e:
            print(f"Error loading {model_type} model: {e}")
            return None, []
            
    return cached_models[model_type], cached_classes

@app.get("/api/classes")
def get_available_classes():
    if not os.path.exists(DATASET_DIR):
        # Fallback to model classes if loaded, or soil descriptions
        _, classes = get_model("svm")
        if classes:
            return {"classes": sorted(classes)}
        return {"classes": sorted(list(SOIL_DESCRIPTIONS.keys()))}
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
        "total_epochs": 3,
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
async def predict_image(file: UploadFile = File(...), model_type: str = Form("svm")):
    model, classes = get_model(model_type)
    if model is None:
        raise HTTPException(status_code=400, detail=f"Model '{model_type}' is not trained or loaded. Please train the models first.")
        
    try:
        # Load and preprocess image
        image_content = await file.read()
        image = Image.open(io.BytesIO(image_content)).convert("RGB")
        
        # Extract features
        feats = extract_features(image, img_size=(32, 32))
        img_batch = np.expand_dims(feats, axis=0) # Add batch dimension (1, 3102)
        
        # Run prediction
        predictions = model.predict_proba(img_batch)[0]
        max_idx = int(np.argmax(predictions))
        predicted_class = classes[max_idx]
        confidence = float(predictions[max_idx]) * 100
        
        # Get matching probabilities for all classes
        probabilities = {classes[i]: float(predictions[i]) * 100 for i in range(len(classes))}
        
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

# Serve static files for frontend UI
static_path = os.path.join(BASE_DIR, "static")
os.makedirs(static_path, exist_ok=True)
app.mount("/", StaticFiles(directory=static_path, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)
