# server.py — backend for frog chorus app

import asyncio
import csv
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import librosa
import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocketDisconnect
from pydantic import BaseModel

from frog_utils import (
    SR,
    make_mel_spec_for_array,
    compute_aux_features_for_array,
    idx2label,
)

# -------------------------------------------------------------------
# Paths & config
# -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "frog_mel_aux_model_v2.h5"
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

OBS_CSV = ROOT / "citizen_observations.csv"

# sliding window params (must match front-end story)
WINDOW_SEC = 1.5
HOP_SEC = 0.5

# -------------------------------------------------------------------
# Load model
# -------------------------------------------------------------------

print("Loading model from:", MODEL_PATH)
model = tf.keras.models.load_model(str(MODEL_PATH))

# -------------------------------------------------------------------
# FastAPI app + CORS
# -------------------------------------------------------------------

app = FastAPI(title="Frog Chorus Backend")

# allow file:// page or localhost frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def classify_file(filepath: Path):
    """
    Run the CNN on a whole recording once, for the upload panel.
    """
    y, sr = librosa.load(str(filepath), sr=SR, mono=True)
    spec = make_mel_spec_for_array(y)
    aux = compute_aux_features_for_array(y)

    spec_in = np.expand_dims(spec, 0)          # [1, T, F, 1]
    aux_in = np.expand_dims(aux.astype("float32"), 0)  # [1, 2]

    probs = model.predict([spec_in, aux_in], verbose=0)[0]
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    label = idx2label[top_idx]

    return label, top_prob, probs.tolist()


async def stream_file_segments(filepath: Path, ws: WebSocket):
    """
    Walk through a wav file in sliding windows and send predictions as JSON.

    Runs ONCE from start to end so it can stay in sync with the audio playback
    on the front-end.
    """
    y, sr = librosa.load(str(filepath), sr=SR, mono=True)
    total = len(y)
    win_samples = int(WINDOW_SEC * SR)
    hop_samples = int(HOP_SEC * SR)

    print("Streaming file once:", filepath)
    print("Duration (s):", total / SR)

    t = 0.0
    i = 0

    while True:
        start = i * hop_samples
        end = start + win_samples
        if start >= total:
            break

        seg = y[start:end]
        if len(seg) < int(0.3 * SR):
            break

        # feature extraction
        spec = make_mel_spec_for_array(seg)
        aux = compute_aux_features_for_array(seg)

        spec_in = np.expand_dims(spec, 0)
        aux_in = np.expand_dims(aux.astype("float32"), 0)

        probs = model.predict([spec_in, aux_in], verbose=0)[0]
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        label = idx2label[top_idx]

        msg = {
            "recording_id": filepath.name,
            "start_time": round(t, 3),
            "end_time": round(t + WINDOW_SEC, 3),
            "top_label": label,
            "top_prob": top_prob,
            "probs": probs.tolist(),
        }

        await ws.send_text(json.dumps(msg))

        # simulate real time
        await asyncio.sleep(HOP_SEC)
        t += HOP_SEC
        i += 1

    print("Finished streaming file:", filepath)


class ValidationPayload(BaseModel):
    recording_id: str
    species_label: str
    user_answer: str   # "yes" or "no"
    top_prob: float | None = None


def append_observation(row: ValidationPayload):
    """
    Append a citizen validation row to CSV.
    """
    file_exists = OBS_CSV.exists()
    with OBS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                ["timestamp", "recording_id", "species_label", "user_answer", "top_prob"]
            )
        writer.writerow(
            [
                datetime.now().isoformat(),
                row.recording_id,
                row.species_label,
                row.user_answer,
                row.top_prob if row.top_prob is not None else "",
            ]
        )


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Frog server running",
        "model_path": str(MODEL_PATH),
        "upload_dir": str(UPLOAD_DIR),
    }


@app.post("/api/upload_frog")
async def upload_frog(file: UploadFile = File(...)):
    """
    Receive an uploaded recording, save it, run a single global prediction,
    and return top-1 info + recording_id.
    """
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # create a unique filename in uploads/
    suffix = Path(file.filename).suffix or ".wav"
    save_name = f"{int(datetime.now().timestamp())}_{uuid4().hex}{suffix}"
    save_path = UPLOAD_DIR / save_name

    contents = await file.read()
    save_path.write_bytes(contents)

    try:
        top_label, top_prob, probs = classify_file(save_path)
    except Exception as e:
        print("Error classifying upload:", e)
        raise HTTPException(status_code=500, detail="Error analysing recording")

    return {
        "recording_id": save_path.name,
        "top_label": top_label,
        "top_prob": top_prob,
        "probs": probs,
    }


@app.post("/api/validate_observation")
async def validate_observation(payload: ValidationPayload):
    """
    Citizen clicks Yes / No. Save their validation to CSV.
    """
    try:
        append_observation(payload)
        return {"status": "ok"}
    except Exception as e:
        print("Error writing observation:", e)
        raise HTTPException(status_code=500, detail="Could not save observation")


@app.websocket("/ws/frogs")
async def frog_ws(ws: WebSocket):
    """
    WebSocket endpoint for streaming sliding-window predictions
    for a specific uploaded recording.

    Client must pass ?recording_id=<filename> in the URL.
    """
    await ws.accept()
    print("WebSocket client connected")

    try:
        params = dict(ws.query_params)
        rec_id = params.get("recording_id")

        if not rec_id:
            await ws.send_text(json.dumps({"error": "No recording_id provided"}))
            return

        filepath = UPLOAD_DIR / rec_id
        if not filepath.exists():
            await ws.send_text(
                json.dumps({"error": f"Recording not found on server: {rec_id}"})
            )
            return

        await stream_file_segments(filepath, ws)

    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print("Error in WebSocket:", e)
    finally:
        try:
            await ws.close()
        except Exception:
            pass
        print("WebSocket closed")


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting frog server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
