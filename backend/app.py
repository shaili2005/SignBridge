from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np

from model import SignRecognizer
from storage import fetch_recent_predictions, initialize_database, save_prediction

app = FastAPI(title="SignBridge API")
recognizer = SignRecognizer()
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

initialize_database()
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/app")
def app_page():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/style.css")
def frontend_styles():
    return FileResponse(FRONTEND_DIR / "style.css")


@app.get("/script.js")
def frontend_script():
    return FileResponse(FRONTEND_DIR / "script.js")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "supported_signs": [
            "HELLO",
            "PLEASE",
            "YES",
            "NO",
            "OK",
            "PEACE",
            "THUMBS_UP",
            "THUMBS_DOWN",
            "I_LOVE_YOU",
        ],
    }


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
