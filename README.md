# NeuroID – repaired EEG authentication project

## What was repaired
- EEG processing now uses 2-second windows and falls back to fixed-length windows when an EDF has no `T2` annotations, instead of crashing with “Could not find any of the events you specified.”
- Authentication responses remove the NumPy feature matrix before returning JSON, fixing FastAPI response serialization errors.
- The API allows the Vite dev server on ports 5173 and 5174.
- Failed enrollment removes files uploaded during that failed attempt and reports the underlying exception type.
- The bundled user database and model were reset because the previous model was trained with the old feature shape. Create accounts and enroll again using this version.

## Run on macOS
Open Terminal in this folder (the folder containing `run_backend.sh` and `run_frontend.sh`).

### Backend (Terminal 1)
```bash
cd backend
python3 -m venv ../.venv
source ../.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

### Frontend (Terminal 2)
```bash
cd frontend
npm install
npm run dev
```
Open the local address Vite prints (usually `http://localhost:5173`).

## First-time use
1. Create an Operator ID in the web app.
2. Go to Signal Authentication → Enroll EEG and select at least five compatible `.edf` recordings for that operator.
3. To train the SVM model, enroll at least one additional operator using recordings from that operator. A model is only trained after files from at least two operator IDs have been enrolled.
4. Authenticate using one EDF recording from an enrolled operator.

The included demo recordings are PhysioNet-style sample EEG recordings in `data/demo`. They are for software testing and demonstration, not medical diagnosis or proof of real-world biometric security.

## Verification performed in the build environment
- Python compilation check for `backend/app.py` and `backend/eeg_engine.py`: passed.
- Processed included EDF recordings: passed; output feature shape `(15, 321)` and frequency-band metrics were produced.
- No-annotation fallback: passed; fixed-length epochs were created.
- Trained an SVM from the included sample recordings, authenticated a sample recording, and JSON-serialized the response: passed.
- Generated raw EEG, epoch, and PSD plot files: passed.

Limitations: I could not verify this project on your specific Mac/browser or run a complete browser-driven signup/upload workflow here. EEG similarity/confidence is a model score, not a guaranteed identity match. Results depend on recording quality, compatible EDF formats, and appropriate enrollment data.
