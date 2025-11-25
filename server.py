# server.py  — frog streaming server (loops the test file)

import asyncio
import json
from pathlib import Path

import numpy as np
import librosa
import tensorflow as tf
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
import uvicorn

from frog_utils import (
    SR,
    make_mel_spec_for_array,
    compute_aux_features_for_array,
    idx2label,
)

# ===========================
# CONFIG
# ===========================

# Root where your project data lives
ROOT = Path(r"D:\Frog_Calls_Raven")

# Pick a test file to simulate “live” audio
# (right now: first .wav found in Testing_Call_Recordings)
TEST_FILE = next((ROOT / "Testing_Call_Recordings").rglob("*.wav"))
print("Using test file:", TEST_FILE)

window_sec = 1.5   # same as your sliding window
hop_sec = 0.5      # same as your hop

# ===========================
# LOAD MODEL
# ===========================

# ⬇⬇⬇  IMPORTANT: use the new, better model
MODEL_PATH = "frog_mel_aux_model_v2.h5"
print("Loading model from:", MODEL_PATH)
model = tf.keras.models.load_model("frog_mel_aux_model_v2.h5")


app = FastAPI()


# ===========================
# SIMPLE ROOT ROUTE (for sanity check)
# ===========================

@app.get("/")
def root():
    return {
        "message": "Frog server running!",
        "test_file": str(TEST_FILE),
        "window_sec": window_sec,
        "hop_sec": hop_sec,
        "model_path": MODEL_PATH,
    }


# ===========================
# STREAMING HELPER
# ===========================

async def stream_file_segments(filepath: Path, ws: WebSocket):
    """
    Walk through a wav file in sliding windows and send predictions as JSON.

    This version LOOPS the file forever:
    after it reaches the end, it waits briefly and starts again.
    """
    # load once
    y, sr = librosa.load(str(filepath), sr=SR, mono=True)
    total = len(y)
    win_samples = int(window_sec * SR)
    hop_samples = int(hop_sec * SR)

    print("Audio duration (s):", total / SR)

    # outer loop: repeat forever until client disconnects
    while True:
        t = 0.0
        i = 0

        # inner loop: walk through the whole file once
        while True:
            start = i * hop_samples
            end = start + win_samples
            if start >= total:
                break

            seg = y[start:end]
            if len(seg) < int(0.3 * SR):
                break

            # --- feature extraction (same as notebook helpers) ---
            spec = make_mel_spec_for_array(seg)              # [T, F, 1]
            aux = compute_aux_features_for_array(seg)        # [2]

            spec_in = np.expand_dims(spec, 0)                # [1, T, F, 1]
            aux_in = np.expand_dims(aux.astype("float32"), 0)

            # --- model prediction ---
            probs = model.predict([spec_in, aux_in], verbose=0)[0]
            top_idx = int(np.argmax(probs))
            top_prob = float(probs[top_idx])
            label = idx2label[top_idx]

            msg = {
                "recording_id": "live_demo",
                "start_time": round(t, 3),
                "end_time": round(t + window_sec, 3),
                "top_label": label,
                "top_prob": top_prob,
                "probs": probs.tolist(),
            }

            # Send to the browser; if client disconnects, this will raise
            await ws.send_text(json.dumps(msg))

            # simulate real time
            await asyncio.sleep(hop_sec)
            t += hop_sec
            i += 1

        # finished one pass through the file; pause a bit then loop
        await asyncio.sleep(1.0)  # tiny breathing space between loops


# ===========================
# WEBSOCKET ENDPOINT
# ===========================

@app.websocket("/ws/frogs")
async def frog_ws(ws: WebSocket):
    await ws.accept()
    print("WebSocket client connected")
    try:
        await stream_file_segments(TEST_FILE, ws)
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print("Error in stream:", e)
    finally:
        await ws.close()
        print("WebSocket closed")


# ===========================
# ENTRY POINT
# ===========================

if __name__ == "__main__":
    print("Starting frog server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
