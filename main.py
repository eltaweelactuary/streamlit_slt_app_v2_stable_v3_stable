import os
import time
import tempfile
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import json

# Import the core engine
from sign_language_core import SignLanguageCore, DigitalHumanRenderer

app = FastAPI(title="Konecta SLT Enterprise API")

# Initialize Core Engine
WRITABLE_BASE = os.path.join(tempfile.gettempdir(), "slt_persistent_storage")
os.makedirs(WRITABLE_BASE, exist_ok=True)
core = SignLanguageCore(data_dir=WRITABLE_BASE)

# Ensure model is trained/loaded
if not core.load_core():
    print("🚀 Model not found, initializing auto-train sequence...")
    # In a real enterprise app, we'd have a pre-trained model. 
    # For this POC, we can trigger training if benchmarks exist.
    pass

# Mount static files for the frontend
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

class TranslateRequest(BaseModel):
    text: str

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.post("/api/predict")
async def predict_video(file: UploadFile = File(...)):
    """Handles video file uploads for sentence recognition"""
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name
        
        labels, confidence = core.predict_sentence(temp_path)
        os.unlink(temp_path)
        
        if labels:
            # Deduplicate
            result_labels = []
            for l in labels:
                if not result_labels or result_labels[-1] != l:
                    result_labels.append(l)
            
            return {
                "success": True,
                "labels": result_labels,
                "confidence": float(confidence),
                "sentence": " ".join(result_labels)
            }
        return {"success": False, "error": "No signs detected"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/capture")
async def analyze_frame(file: UploadFile = File(...)):
    """Handles single frame capture from camera"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        import mediapipe as mp
        mp_holistic = mp.solutions.holistic
        with mp_holistic.Holistic(min_detection_confidence=0.5, refine_face_landmarks=True) as holistic:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = holistic.process(img_rgb)
            landmarks = core.extract_frame_features(results)
            
            if sum(landmarks) != 0:
                label, confidence = core.predict_from_landmarks(np.array(landmarks))
                if label and confidence > 50:
                    return {"success": True, "label": label, "confidence": float(confidence)}
            
        return {"success": False, "error": "Sign not recognized"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/translate")
async def translate_text(request: TranslateRequest):
    """Translates text to Digital Human DNA (JSON)"""
    try:
        words = request.text.lower().split()
        dna_json = core.get_words_dna_json(words)
        if dna_json:
            return {"success": True, "dna": dna_json}
        return {"success": False, "error": "Vocabulary not available for these words"}
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
