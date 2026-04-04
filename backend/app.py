from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp

app = FastAPI()

# ✅ Enable CORS (IMPORTANT for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_face = mp.solutions.face_detection
face = mp_face.FaceDetection()

# 🎯 Gesture Detection Logic (basic)
def detect_gesture(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # 👉 Get all required landmarks
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]

            index_base = hand_landmarks.landmark[5]
            middle_base = hand_landmarks.landmark[9]
            ring_base = hand_landmarks.landmark[13]
            pinky_base = hand_landmarks.landmark[17]

            explanation = []

            # 🔥 NORMALIZATION (IMPORTANT FIX)
            # Distance reference (wrist to middle finger base)
            wrist = hand_landmarks.landmark[0]
            ref_dist = abs(wrist.y - middle_base.y) + 1e-6  # avoid division by 0

            # Normalize finger positions
            index_up = (index_base.y - index_tip.y) / ref_dist
            middle_up = (middle_base.y - middle_tip.y) / ref_dist
            ring_up = (ring_base.y - ring_tip.y) / ref_dist
            pinky_up = (pinky_base.y - pinky_tip.y) / ref_dist

            # -------------------------
            # ✅ HELLO → OPEN PALM
            # -------------------------
            if (
                index_up > 0.4 and
                middle_up > 0.4 and
                ring_up > 0.4 and
                pinky_up > 0.4
            ):
                explanation.append("All fingers extended")
                return "HELLO", 0.95, ", ".join(explanation)

            # -------------------------
            # ✅ YES → ONLY INDEX UP
            # -------------------------
            if (
                index_up > 0.4 and
                middle_up < 0.2 and
                ring_up < 0.2 and
                pinky_up < 0.2
            ):
                explanation.append("Index finger raised")
                explanation.append("Other fingers folded")
                return "YES", 0.9, ", ".join(explanation)

            # -------------------------
            # ✅ NO → CLOSED FIST
            # -------------------------
            if (
                index_up < 0.2 and
                middle_up < 0.2 and
                ring_up < 0.2 and
                pinky_up < 0.2
            ):
                explanation.append("All fingers folded")
                return "NO", 0.85, ", ".join(explanation)

    return "No Gesture", 0.0, "Hand not clearly detected"

# 🚀 API Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gesture, confidence, explanation = detect_gesture(frame)

    return {
        "gesture": gesture,
        "confidence": confidence,
        "explanation": explanation
    }


# Test route
@app.get("/")
def home():
    return {"message": "Backend Running ✅"}