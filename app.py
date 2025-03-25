import io
import os
import cv2
import time
import base64
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from tempfile import NamedTemporaryFile

import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Constants
MODEL_PATH = os.getenv("MODEL_PATH", "final.pt")  # Get from env or use default
CONFIDENCE_THRESHOLD = 0.25  # Minimum detection confidence

# Initialize FastAPI app
app = FastAPI(
    title="Car Damage Detection API",
    description="API for detecting car damage using YOLOv11 model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Response models
class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    box_coordinates: List[float]

class PredictionResponse(BaseModel):
    processed_image: str  # Base64 encoded image
    inference_time: float
    total_objects: int
    detections: List[DetectionResult]
    class_counts: Dict[str, int]

# Load model function that exactly mirrors inference.py approach
def load_model():
    print(f"Loading YOLO model from {MODEL_PATH}...")
    try:
        # Try loading with Ultralytics YOLO (works for YOLOv8+)
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
        model_type = "ultralytics"
        print("Model loaded with Ultralytics YOLO")
        return model, model_type
    except Exception as e:
        print(f"Error loading with Ultralytics: {str(e)}")
        try:
            # Fallback to PyTorch Hub (YOLOv5)
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
            model_type = "pytorch"
            print("Model loaded with PyTorch Hub (YOLOv5)")
            return model, model_type
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None

# Load model on startup
model, model_type = load_model()
print(f"Model loaded: {model is not None}, Type: {model_type}")

@app.get("/")
async def root():
    return {"message": "Welcome to Car Damage Detection API", "status": "online"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model_loaded": model is not None, 
        "model_type": model_type,
        "classes": model.names if model is not None and hasattr(model, "names") else None
    }

@app.post("/detect", response_model=PredictionResponse)
async def detect_damage(
    file: UploadFile = File(...),
    confidence: float = Form(CONFIDENCE_THRESHOLD)
):
    # Declare globals at the beginning of the function
    global model, model_type
    
    # Check if model is loaded
    if model is None:
        # Try loading model again
        loaded_model, loaded_model_type = load_model()
        if loaded_model is None:
            raise HTTPException(status_code=500, detail="Model not loaded properly")
        else:
            model = loaded_model
            model_type = loaded_model_type
    
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")
    
    try:
        # Read image content
        content = await file.read()
        
        # Convert to OpenCV format
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Store original image for drawing
        orig_img = img.copy()
        height, width = img.shape[:2]
        
        # Convert BGR to RGB (YOLO expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run inference
        start_time = time.time()
        
        # Perform detection
        detections = []
        try:
            if model_type == "ultralytics":
                # YOLOv11 inference with ultralytics
                results = model(img_rgb, conf=confidence)
                
                # Process results
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence_score = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        try:
                            class_name = result.names[class_id]
                        except:
                            class_name = f"class_{class_id}"
                        
                        detections.append({
                            'box': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': confidence_score, 
                            'class_id': class_id,
                            'class_name': class_name
                        })
            elif model_type == "pytorch":
                # YOLOv5 inference (using the same code as in inference.py)
                results = model(img_rgb)
                predictions = results.xyxy[0].cpu().numpy()
                
                for x1, y1, x2, y2, conf, cls_id in predictions:
                    if conf >= confidence:
                        try:
                            class_name = model.names[int(cls_id)]
                        except:
                            class_name = f"class_{int(cls_id)}"
                            
                        detections.append({
                            'box': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(conf),
                            'class_id': int(cls_id),
                            'class_name': class_name
                        })
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
        
        inference_time = time.time() - start_time
        
        # Draw results on the original image
        for det in detections:
            # Get detection info
            box = det['box']
            x1, y1, x2, y2 = [int(coord) for coord in box]
            class_id = det['class_id']
            class_name = det['class_name']
            confidence_score = det['confidence']
            
            # Generate a color based on class id
            color = (
                hash(str(class_id)) % 256,
                (hash(str(class_id)) * 2) % 256,
                (hash(str(class_id)) * 3) % 256
            )
            
            # Draw bounding box
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label = f"{class_name}: {confidence_score:.2f}"
            
            # Draw label background
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(orig_img, (x1, y1 - 20), (x1 + text_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(orig_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Convert result image to base64
        _, buffer = cv2.imencode('.jpg', orig_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Calculate class counts
        class_counts = {}
        for det in detections:
            class_name = det['class_name']
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1
                
        # Format detection results for response
        detection_results = []
        for det in detections:
            detection_results.append(DetectionResult(
                class_name=det['class_name'],
                confidence=det['confidence'],
                box_coordinates=det['box']
            ))
        
        # Prepare the response
        response = PredictionResponse(
            processed_image=img_base64,
            inference_time=inference_time,
            total_objects=len(detections),
            detections=detection_results,
            class_counts=class_counts
        )
        
        return response
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 
