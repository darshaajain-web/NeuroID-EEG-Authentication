"""Register two demo operators from the bundled PhysioNet-derived EDF files."""
from pathlib import Path
import shutil
from app import UPLOADS, PLOTS, USERS_FILE, MODEL_FILE, save_users
from eeg_engine import process_eeg, train_model

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "demo" / "register"
users = {}
for label, username in enumerate(["DemoUser1", "DemoUser2"], 1):
    files = sorted((SOURCE / username.lower().replace("demouser", "demo_user_")).glob("*.edf"))
    # folder names are demo_user_1 and demo_user_2
    if not files:
        raise SystemExit(f"No demo files found for {username}")
    for i, src in enumerate(files):
        shutil.copy2(src, UPLOADS / f"{username}_{i}_{src.name}")
    users[username] = {"email": "demo@example.com", "label": label, "files": len(files)}
    out = PLOTS / username
    out.mkdir(parents=True, exist_ok=True)
    for src in files:
        process_eeg(src, plot_dir=out, save_plots=True)
save_users(users)
train_model(users, MODEL_FILE, UPLOADS)
print("Demo model ready for DemoUser1 and DemoUser2")
