
# 🐸 FrogCallsAI — Bush Frog Species Classifier & Live Visualization

Real-time acoustic species detection for Western Ghats bush frogs using a CNN audio model + auxiliary features.
Includes data preprocessing notebook, trained model, WebSocket inference API, and a radial D3 visualization.

---

## 🎯 Features

| Module                  | Description                                                                 |
| ----------------------- | --------------------------------------------------------------------------- |
| **Model Training**      | Jupyter notebook that segments audio, extracts mel-spectrograms, trains CNN |
| **Inference Backend**   | FastAPI WebSocket server streaming predictions in sliding windows           |
| **Frontend UI**         | Fully interactive radial soundscape visualization built with D3.js          |
| **Live Audio Analysis** | Test file loops and displays species presence in real-time                  |

Supported Species:

* **Raven_amboli** – Amboli Bush Frog
* **Raven_coorg** – Coorg Yellow Bush Frog
* **Raven_knob** – Knob-Handed Bush Frog

---

## 📁 Project Structure

```
FROG_LIVE_VIS/
│
├── server.py                      # WebSocket streaming server
├── frog_utils.py                  # Preprocessing & feature extraction
├── frog_mel_aux_model_v2.h5       # Final trained model (~90% val accuracy)
├── clean_aux_scaler.joblib        # Scaler used during training
├── label_mapping.json             # Species → index mapping (ordered list)
├── test_frog_recording.wav        # Sample audio looped for live demo
│
├── frog_chorus_radial_xfiles.html # D3 visualization frontend
│
└── training/
    └── frog_clean_pipeline.ipynb  # Full dataset cleaning + training pipeline
```

⚠️ Raw dataset (`species/`, `Testing_Call_Recordings/`) excluded due to size.

---

## 🧠 Model Summary

Deep CNN + auxiliary features (`peak_log`, `active_s`)
Training audio resampled → segmented → denoised → mel-spectrogram (374 frames × 64 mel-bins)

* Loss: Sparse categorical cross-entropy
* Validation Accuracy: **~90%**
* Balanced class training via oversampling
* Peak frequency helps distinguish similar calls ⚡

---

## 🚀 How to Run (Local Demo)

### 1️⃣ Start server (backend)

```bash
cd FROG_LIVE_VIS
python -m uvicorn server:app --reload --port 8000
```

Leave this **terminal running**.

### 2️⃣ Open the UI (frontend)

Open this file in Chrome:

```
FROG_LIVE_VIS/frog_chorus_radial_xfiles.html
```

You should see:

✔ Species arcs animating
✔ Dots pulsing with predicted windows
✔ Confidence & dominance stats updating
✔ Audio can be played in sync

---

## 💡 Visualization + UX Highlights

* Time wraps around a circle (last 60s shown)
* Each **ring = species**
* Each **arc = detection window**
* Glow & dot size encode **confidence & peak frequency**
* Summary panel interprets the “chorus story”

Designed using a thematic **X-Files night aesthetic**
to match field acoustics culture 🌌

---

## 🧪 Re-Training (if needed later)

Inside notebook:

```python
SHOULD_TRAIN = True
```

Then run Cells 1→10 to regenerate the model and scaler.

Output files will update automatically:

* `frog_mel_aux_model_v2.h5`
* `clean_aux_scaler.joblib`
* `label_mapping.json`

---

## 🔮 Future Work

* Expand to more species from Western Ghats
* Deploy backend online (cloud / edge device)
* Accept microphone input for **true live monitoring**
* Add "unknown species" rejection classifier

---

## 🧑‍🎓 Credits

**Author**: Gayatri Jadhav
Srishti Manipal Institute of Art, Design & Technology
Project: Acoustic Indian Anuran Species Classifier

---

## 📝 Notes for Evaluators

✔ No training required — model + scaler baked in
✔ Fully working **real-time** prediction demo
✔ Clean modular code for future extension

If any setup issues occur:
check Python terminal for errors → model shapes must match `TARGET_FRAMES = 374` (already set)

