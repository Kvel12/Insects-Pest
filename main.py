"""
Insect Pest Classification API
FastAPI service for predicting insect pests in crops
"""

import os
import io
import json
import base64
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google.cloud import storage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

# Parámetros del modelo
IMG_SIZE = (224, 224)  # ResNet50 input size
MODEL_PATH = "/app/model/resnet50_ip102_final.keras"
PESTS_CSV_PATH = "/app/data/plagas_cultivos_completo.csv"

# Cloud Storage
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "datasets_bucket_01")
GCS_MODEL_BLOB = os.getenv("GCS_MODEL_BLOB", "models/trained/resnet50_ip102_final.keras")
GCS_PESTS_BLOB = os.getenv("GCS_PESTS_BLOB", "results/plagas_cultivos_completo.csv")

# Google Drive IDs (fallback)
MODEL_GDRIVE_ID = os.getenv("MODEL_GDRIVE_ID", "")
PESTS_GDRIVE_ID = os.getenv("PESTS_GDRIVE_ID", "")

# ============================================================================
# INICIALIZACIÓN DE FASTAPI
# ============================================================================

app = FastAPI(
    title="Insect Pest Classification API",
    description="AI-powered insect pest detection for agricultural crop protection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODELOS PYDANTIC
# ============================================================================

class TopPrediction(BaseModel):
    """Individual prediction with class name and confidence"""
    class_id: int = Field(..., description="ID de la clase (0-101)")
    class_name: str = Field(..., description="Nombre de la plaga")
    confidence: float = Field(..., description="Confianza (0-1)")

class PestInfo(BaseModel):
    """Detailed pest information"""
    nombre: str = Field(..., description="Nombre común")
    nombre_cientifico: str = Field(..., description="Nombre científico")
    cultivo_principal: str = Field(..., description="Cultivo principal afectado")
    cultivos_afectados: str = Field(..., description="Lista de cultivos afectados")
    categoria: str = Field(..., description="Categoría de cultivo")
    severidad: str = Field(..., description="Nivel de severidad")
    tipo_daño: str = Field(..., description="Tipo de daño causado")
    region: str = Field(..., description="Regiones geográficas")

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predicted_class_id: int = Field(..., description="ID de la clase predicha")
    predicted_class_name: str = Field(..., description="Nombre de la plaga predicha")
    confidence: float = Field(..., description="Confianza de la predicción (0-1)")
    top_predictions: List[TopPrediction] = Field(..., description="Top 5 predicciones")
    pest_info: Optional[PestInfo] = Field(None, description="Información detallada de la plaga")
    reference_image_base64: Optional[str] = Field(None, description="Imagen procesada en base64")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    pests_data_loaded: bool
    total_classes: int
    version: str
    download_source: Optional[str] = None

# ============================================================================
# ESTADO GLOBAL
# ============================================================================

class ModelState:
    """Global state for model and data"""
    model: Optional[keras.Model] = None
    classes_dict: Optional[Dict[int, str]] = None  # {0: 'nombre_plaga', 1: ...}
    pests_df: Optional[pd.DataFrame] = None
    download_source: str = "not_loaded"
    
model_state = ModelState()

# ============================================================================
# FUNCIONES AUXILIARES - DESCARGA
# ============================================================================

def download_from_gcs(bucket_name: str, blob_name: str, destination: Path) -> bool:
    """Download from Cloud Storage"""
    try:
        logger.info(f"Downloading {blob_name} from GCS")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(destination))
        
        logger.info(f"GCS download success")
        return True
    except Exception as e:
        logger.error(f"GCS error: {str(e)}")
        return False

def download_from_google_drive(file_id: str, destination: Path) -> bool:
    """Download from Google Drive"""
    try:
        import gdown
        logger.info(f"Downloading from Google Drive: {file_id}")
        
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(destination), quiet=False)
        
        logger.info(f"✅ Google Drive download success")
        return True
    except Exception as e:
        logger.error(f"Google Drive error: {str(e)}")
        return False

# ============================================================================
# CARGA DE MODELO Y DATOS
# ============================================================================

def load_model_and_data():
    """Load model and pests data at startup"""
    try:
        # ===== CARGAR MODELO =====
        model_path = Path(MODEL_PATH)
        if not model_path.exists():
            logger.info("Model not found, downloading...")
            success = download_from_gcs(GCS_BUCKET_NAME, GCS_MODEL_BLOB, model_path)
            if success:
                model_state.download_source = "cloud_storage"
            else:
                if MODEL_GDRIVE_ID:
                    success = download_from_google_drive(MODEL_GDRIVE_ID, model_path)
                    if success:
                        model_state.download_source = "google_drive"
            
            if not success:
                raise Exception("Failed to download model")
        else:
            model_state.download_source = "local_cache"
        
        # Cargar modelo
        logger.info("Loading model...")
        model_state.model = keras.models.load_model(str(model_path))
        logger.info(f"Model loaded: {model_state.model.count_params():,} params")
        
        # Obtener número de clases
        num_classes = model_state.model.output_shape[-1]
        logger.info(f"Model expects {num_classes} classes")
        
        # ===== CARGAR CSV DE PLAGAS =====
        pests_path = Path(PESTS_CSV_PATH)
        if not pests_path.exists():
            logger.info("Pests CSV not found, downloading...")
            success = download_from_gcs(GCS_BUCKET_NAME, GCS_PESTS_BLOB, pests_path)
            if not success and PESTS_GDRIVE_ID:
                success = download_from_google_drive(PESTS_GDRIVE_ID, pests_path)
            
            if not success:
                logger.warning("Could not download pests CSV")
        
        if pests_path.exists():
            model_state.pests_df = pd.read_csv(str(pests_path), index_col=0)
            logger.info(f"Pests data loaded: {len(model_state.pests_df)} records")
            
            # Crear diccionario de clases desde el CSV
            model_state.classes_dict = {}
            for idx, row in model_state.pests_df.iterrows():
                model_state.classes_dict[int(idx)] = row['nombre']
            
            logger.info(f"Classes dictionary created: {len(model_state.classes_dict)} classes")
        else:
            # Si no hay CSV, crear diccionario básico
            logger.warning("Creating basic class dictionary without pest data")
            model_state.classes_dict = {i: f"Pest Class {i}" for i in range(num_classes)}
        
        # VERIFICAR
        if len(model_state.classes_dict) != num_classes:
            logger.warning(f"Class count mismatch: model has {num_classes}, dict has {len(model_state.classes_dict)}")
        
        logger.info(f"✅ API ready with {num_classes} classes")
        return True
        
    except Exception as e:
        logger.error(f"Load error: {str(e)}")
        return False

def preprocess_image(image_bytes: bytes) -> tuple:
    """Preprocess image for ResNet50 prediction"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode image")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    img_resized = cv2.resize(img, IMG_SIZE)
    
    # Preprocess for ResNet50
    img_array = keras.applications.resnet50.preprocess_input(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, img_resized

def image_to_base64(image_array: np.ndarray) -> str:
    """Convert image to base64"""
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', image_bgr)
    return base64.b64encode(buffer).decode('utf-8')

def get_pest_info(class_id: int) -> Optional[PestInfo]:
    """Get detailed pest information"""
    if model_state.pests_df is None or class_id not in model_state.pests_df.index:
        return None
    
    try:
        row = model_state.pests_df.loc[class_id]
        return PestInfo(
            nombre=str(row['nombre']),
            nombre_cientifico=str(row['nombre_cientifico']),
            cultivo_principal=str(row['cultivo_principal']),
            cultivos_afectados=str(row['cultivos_afectados_texto']),
            categoria=str(row['categoria']),
            severidad=str(row['severidad']),
            tipo_daño=str(row['tipo_daño']),
            region=str(row['region'])
        )
    except Exception as e:
        logger.warning(f"Error getting pest info for class {class_id}: {str(e)}")
        return None

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting up...")
    success = load_model_and_data()
    if success:
        logger.info("Ready")
    else:
        logger.error("Failed to load")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Insect Pest Classification API",
        "version": "1.0.0",
        "model": "ResNet50",
        "dataset": "IP102",
        "classes": 102,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check"""
    return HealthResponse(
        status="healthy" if model_state.model else "unhealthy",
        model_loaded=model_state.model is not None,
        pests_data_loaded=model_state.pests_df is not None,
        total_classes=len(model_state.classes_dict) if model_state.classes_dict else 0,
        version="1.0.0",
        download_source=model_state.download_source
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    include_reference_image: bool = True,
    top_k: int = 5
):
    """Predict insect pest from image"""
    if model_state.model is None:
        raise HTTPException(503, "Model not loaded")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "Invalid file type. Must be an image.")
    
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        img_array, img_resized = preprocess_image(image_bytes)
        
        # Predict
        predictions = model_state.model.predict(img_array, verbose=0)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx])
        predicted_class_name = model_state.classes_dict.get(predicted_idx, f"Unknown Class {predicted_idx}")
        
        # Top K predictions
        top_k = min(top_k, len(predictions[0]))
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        top_predictions = [
            TopPrediction(
                class_id=int(i),
                class_name=model_state.classes_dict.get(int(i), f"Class {i}"),
                confidence=float(predictions[0][i])
            )
            for i in top_indices
        ]
        
        # Get pest info
        pest_info = get_pest_info(predicted_idx)
        
        # Generate reference image
        reference_image = image_to_base64(img_resized) if include_reference_image else None
        
        return PredictionResponse(
            predicted_class_id=predicted_idx,
            predicted_class_name=predicted_class_name,
            confidence=confidence,
            top_predictions=top_predictions,
            pest_info=pest_info,
            reference_image_base64=reference_image
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.get("/classes")
async def get_classes():
    """Get all available pest classes"""
    if model_state.classes_dict is None:
        raise HTTPException(503, "Model not loaded")
    
    classes_list = [
        {"class_id": k, "class_name": v} 
        for k, v in sorted(model_state.classes_dict.items())
    ]
    
    return {
        "total_classes": len(classes_list),
        "classes": classes_list
    }

@app.get("/pest/{class_id}")
async def get_pest_details(class_id: int):
    """Get detailed information about a specific pest"""
    if class_id < 0 or class_id > 101:
        raise HTTPException(400, "Class ID must be between 0 and 101")
    
    pest_info = get_pest_info(class_id)
    if pest_info is None:
        raise HTTPException(404, f"No information found for pest class {class_id}")
    
    return {
        "class_id": class_id,
        "class_name": model_state.classes_dict.get(class_id, f"Class {class_id}"),
        "pest_info": pest_info
    }

@app.get("/stats")
async def get_stats():
    """Get statistics about the pest database"""
    if model_state.pests_df is None:
        raise HTTPException(503, "Pests data not loaded")
    
    try:
        stats = {
            "total_pests": len(model_state.pests_df),
            "categories": model_state.pests_df['categoria'].value_counts().to_dict(),
            "severity_levels": model_state.pests_df['severidad'].value_counts().to_dict(),
            "main_crops": model_state.pests_df['cultivo_principal'].value_counts().head(10).to_dict()
        }
        return stats
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")