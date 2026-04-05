# SignBridge
Multimodal and explainable sign language recognition prototype for real-time gesture detection and interpretation.

## Current Scope
- Real-time webcam interface in the browser
- FastAPI backend for prediction
- Multimodal cues from hand, face, and upper-body posture
- Explainable output with confidence and cue summaries
- Readable text output for assistive communication
- SQLite-backed recognition logs

## Project Structure
- `frontend/` contains the browser UI
- `backend/` contains the FastAPI API and recognition logic

## Run Locally
1. Create and activate a Python virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Start the backend from `backend/` using `uvicorn app:app --reload`.
4. Open `frontend/index.html` in the browser.
5. Allow camera access and start live prediction.

## Supported Prototype Signs
- `HELLO`
- `YES`
- `NO`
- `OK`
- `PEACE`
- `THUMBS_UP`
