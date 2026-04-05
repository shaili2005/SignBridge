from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

from model import SignRecognizer
from storage import fetch_recent_predictions, initialize_database, save_prediction

app = FastAPI(title="SignBridge API")
recognizer = SignRecognizer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

initialize_database()


@app.get("/")
def home():
    return {
        "message": "SignBridge backend is running.",
        "supported_signs": ["HELLO", "YES", "NO", "OK", "PEACE", "THUMBS_UP"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image.")

    result = recognizer.predict(frame)
    save_prediction(
        gesture=result.gesture,
        text_output=result.text,
        confidence=result.confidence,
        explanation=result.explanation,
    )

    return {
        "gesture": result.gesture,
        "text": result.text,
        "confidence": result.confidence,
        "explanation": result.explanation,
        "modalities": result.modalities,
    }


@app.get("/logs")
def logs(limit: int = 10):
    return {"items": fetch_recent_predictions(limit=limit)}
