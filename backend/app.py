from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import List

import joblib
import mne
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from eeg_engine import process_eeg, train_model, authenticate, profile_summary

BASE = Path(__file__).resolve().parent
STORAGE = BASE / "storage"
USERS_FILE = STORAGE / "users.json"
MODEL_FILE = STORAGE / "model.joblib"
UPLOADS = STORAGE / "uploads"
PLOTS = STORAGE / "plots"
for p in (STORAGE, UPLOADS, PLOTS):
    p.mkdir(parents=True, exist_ok=True)
if not USERS_FILE.exists():
    USERS_FILE.write_text("{}", encoding="utf-8")

app = FastAPI(title="NeuroID Real EEG API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/plots", StaticFiles(directory=str(PLOTS)), name="plots")


def load_users():
    return json.loads(USERS_FILE.read_text(encoding="utf-8"))


def save_users(users):
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")


def save_upload(upload: UploadFile, target: Path):
    if not upload.filename or not upload.filename.lower().endswith(".edf"):
        raise HTTPException(400, "Only .edf EEG files are accepted.")
    with target.open("wb") as f:
        shutil.copyfileobj(upload.file, f)

@app.get("/api/health")
def health():
    users = load_users()
    return {"status": "online", "users": len(users), "model_ready": MODEL_FILE.exists()}

@app.get("/api/users")
def users():
    data = load_users()
    return {"users": [{"username": k, "email": v.get("email", ""), "label": v.get("label")} for k, v in data.items()]}

@app.post("/api/create-account")
def create_account(username: str = Form(...), email: str = Form("")):
    username = username.strip()
    if not username:
        raise HTTPException(400, "Operator ID is required.")
    users = load_users()
    if username in users:
        raise HTTPException(409, "Operator ID already exists. Choose another ID or sign in.")
    label = max([u.get("label", 0) for u in users.values()] + [0]) + 1
    users[username] = {"email": email.strip(), "label": label, "files": 0, "enrolled": False}
    save_users(users)
    return {"ok": True, "username": username, "label": label, "enrolled": False, "message": "Account created. Enroll at least 5 EEG recordings from the dashboard before biometric login."}

@app.post("/api/register")
def register(username: str = Form(...), email: str = Form(""), files: List[UploadFile] = File(...)):
    username = username.strip()
    if not username:
        raise HTTPException(400, "Operator ID is required.")
    if len(files) < 5:
        raise HTTPException(400, "Enrollment requires at least 5 EEG .edf recordings.")
    users = load_users()
    if username not in users:
        raise HTTPException(404, "Account not found. Create the account first.")
    if email.strip():
        users[username]["email"] = email.strip()
    user_dir = PLOTS / username
    user_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    try:
        start_index = len(list(UPLOADS.glob(f"{username}_*.edf")))
        for i, upload in enumerate(files):
            target = UPLOADS / f"{username}_{start_index+i}_{Path(upload.filename).name}"
            save_upload(upload, target)
            saved.append(target)
        samples = []
        for path in saved:
            result = process_eeg(path, plot_dir=user_dir, save_plots=True)
            samples.append(result["features"])
        total_files = len(list(UPLOADS.glob(f"{username}_*.edf")))
        users[username]["files"] = total_files
        users[username]["enrolled"] = total_files >= 5
        save_users(users)

        # Only train when the stored EEG files actually contain at least two
        # different operators. Account metadata alone must not trigger training.
        enrolled_labels = set()
        for operator, meta in users.items():
            operator_files = list(UPLOADS.glob(f"{operator}_*.edf"))
            if operator_files:
                enrolled_labels.add(int(meta["label"]))

        if len(enrolled_labels) >= 2:
            train_model(users, MODEL_FILE, UPLOADS)
            model_message = "SVM model updated and biometric authentication is ready."
        else:
            model_message = (
                "EEG enrollment saved. This is a valid account, but the SVM needs "
                "EEG enrollment from a second operator before biometric authentication "
                "can be trained. You can continue to the dashboard now."
            )
        return {"ok": True, "username": username, "label": users[username]["label"], "files": total_files, "model_ready": MODEL_FILE.exists(), "message": f"Real EEG enrollment saved. {model_message}"}
    except Exception as exc:
        # Do not leave partially uploaded enrollment files after a failed attempt.
        for path in saved:
            path.unlink(missing_ok=True)
        raise HTTPException(500, f"EEG enrollment failed: {type(exc).__name__}: {exc}")

@app.post("/api/authenticate")
def auth(username: str = Form(...), file: UploadFile = File(...)):
    users = load_users()
    if username not in users:
        raise HTTPException(404, "Operator ID is not registered. Create an account first.")
    if users[username].get("files", 0) <= 0:
        raise HTTPException(400, "This account has no EEG enrollment. Open the dashboard and enroll at least 5 EEG recordings first.")
    if not MODEL_FILE.exists():
        raise HTTPException(400, "Your account is registered and EEG enrollment is saved, but biometric login is not trained yet. Enroll a second operator to train the SVM.")
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        save_upload(file, temp_path)
        result = process_eeg(temp_path, plot_dir=PLOTS / username, save_plots=True)
        result["plot_urls"] = {"raw": f"/plots/{username}/raw-data.png", "epochs": f"/plots/{username}/raw_epochs.png", "psd": f"/plots/{username}/epochs_psd.png"}
        verdict = authenticate(result["features"], users[username]["label"], MODEL_FILE)
        result["authenticated"] = bool(verdict["authenticated"])
        result["predicted_label"] = int(verdict["predicted_label"])
        result["confidence"] = float(verdict["confidence"])
        result["score_type"] = str(verdict["score_type"])
        result["username"] = username
        # Feature matrices are used internally for matching, not sent over JSON.
        result.pop("features", None)
        return result
    except Exception as exc:
        raise HTTPException(500, f"EEG authentication failed: {exc}")
    finally:
        temp_path.unlink(missing_ok=True)

@app.get("/api/profile/{username}")
def profile(username: str):
    users = load_users()
    if username not in users:
        raise HTTPException(404, "User not found")
    return profile_summary(username, users[username], MODEL_FILE)
