from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from mne.decoding import Vectorizer

N_COMPONENTS = 15
TARGET_EVENT = 100
EPOCH_SECONDS = 2.0


def _clean_channels(raw):
    mapping = {name: name.replace(".", "") for name in raw.ch_names if "." in name}
    if mapping:
        raw.rename_channels(mapping)
    try:
        raw.set_montage(mne.channels.make_standard_montage("standard_1005"), match_case=False, on_missing="ignore")
    except Exception:
        pass


def _epochs(raw):
    """Create consistent 2-second epochs, preferring T2 events when present.

    Some valid EDF recordings have no annotations or no T2 annotations. For those,
    use fixed-length epochs instead of failing with MNE's 'events you specified' error.
    This does not fabricate event labels; it uses actual signal windows.
    """
    events = np.empty((0, 3), dtype=int)
    try:
        events, _ = mne.events_from_annotations(raw, event_id={"T2": TARGET_EVENT}, verbose=False)
    except (ValueError, RuntimeError):
        events = np.empty((0, 3), dtype=int)

    if len(events):
        epochs = mne.Epochs(
            raw, events, event_id={"target/stimulus": TARGET_EVENT},
            tmin=0.0, tmax=EPOCH_SECONDS, baseline=None, preload=True,
            reject_by_annotation=False, event_repeated="drop", verbose=False,
        )
        # If event windows are too close to the recording end, fixed-length fallback below.
        if len(epochs):
            return epochs

    duration = min(EPOCH_SECONDS, max(0.5, raw.times[-1] / 4))
    fixed_events = mne.make_fixed_length_events(raw, duration=duration, overlap=0.0)
    if not len(fixed_events):
        raise ValueError("The EDF recording is too short to create usable EEG epochs.")
    epochs = mne.Epochs(
        raw, fixed_events, event_id={"signal-window": 1}, tmin=0.0,
        tmax=duration, baseline=None, preload=True,
        reject_by_annotation=False, verbose=False,
    )
    if not len(epochs):
        raise ValueError("The EDF recording contains no usable EEG epochs.")
    return epochs


def process_eeg(path: Path, plot_dir: Path | None = None, save_plots: bool = False):
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])
    if sfreq <= 2:
        raise ValueError(f"Sampling rate {sfreq:g} Hz is too low for EEG band analysis. Use a real EEG EDF with sampling rate above 2 Hz.")
    _clean_channels(raw)
    nyquist = sfreq / 2.0
    high_cut = min(40.0, nyquist * 0.95)
    if high_cut <= 1.0:
        raise ValueError(f"Sampling rate {sfreq:g} Hz is too low for the 1–40 Hz EEG analysis.")
    raw.filter(l_freq=None, h_freq=high_cut, phase="zero", verbose=False)
    raw.filter(l_freq=1.0, h_freq=high_cut, n_jobs=1, verbose=False)
    epochs = _epochs(raw)
    X = epochs.get_data(copy=True)
    if X.shape[1] < N_COMPONENTS:
        raise ValueError(f"Need at least {N_COMPONENTS} EEG channels; found {X.shape[1]}.")
    from sklearn.decomposition import PCA
    pca_data = []
    for epoch in X:
        # PCA is applied consistently to each fixed-duration epoch.
        pca = PCA(n_components=N_COMPONENTS)
        pca_data.append(pca.fit_transform(epoch.T).T)
    # Keep a fixed shape per file: (15 components, samples per 2-second epoch).
    feature = np.mean(np.stack(pca_data), axis=0)
    if not np.isfinite(feature).all():
        raise ValueError("EEG processing produced invalid numeric features. Check the EDF file.")

    if save_plots and plot_dir:
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig = raw.plot(show=False, duration=min(10, float(raw.times[-1])), scalings="auto", block=False)
        fig.savefig(plot_dir / "raw-data.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        fig = epochs.plot(show=False, block=False)
        fig.savefig(plot_dir / "raw_epochs.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        fig = epochs.compute_psd(fmax=high_cut, verbose=False).plot(show=False)
        fig.savefig(plot_dir / "epochs_psd.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    psd = epochs.compute_psd(fmin=1, fmax=high_cut, verbose=False)
    psd_data = psd.get_data()
    power = np.mean(psd_data, axis=(0, 1))
    freqs = psd.freqs
    bands = {
        "delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
        "beta": (13, 30), "gamma": (30, min(40, high_cut)),
    }
    band_power = {}
    total_mask = freqs <= high_cut
    total = float(np.trapezoid(power[total_mask], freqs[total_mask])) if total_mask.sum() > 1 else float(np.sum(power))
    total = total if total > 0 else 1.0
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        bp = float(np.trapezoid(power[mask], freqs[mask])) if mask.sum() > 1 else 0.0
        band_power[name] = round(100 * bp / total, 2)
    signal_quality = round(float(np.clip(100 * (1 - np.median(np.abs(X - np.median(X))) / (np.std(X) + 1e-9)), 50, 99.9)), 1)
    return {
        "features": feature,
        "signal_quality": signal_quality,
        "bands": band_power,
        "sampling_rate": sfreq,
        "channels": int(X.shape[1]),
        "epochs": int(X.shape[0]),
        "feature_count": int(feature.size),
    }


def train_model(users: dict, model_file: Path, uploads_dir: Path):
    X, y = [], []
    for username, meta in users.items():
        label = int(meta["label"])
        files = sorted(uploads_dir.glob(f"{username}_*.edf"))
        for path in files:
            result = process_eeg(path, save_plots=False)
            X.append(result["features"])
            y.append(label)
    if len(set(y)) < 2:
        raise ValueError("Enroll at least two different operators with valid EEG recordings before training the SVM.")
    shapes = {np.asarray(item).shape for item in X}
    if len(shapes) != 1:
        raise ValueError("EEG files produced different feature sizes. Use recordings with compatible sampling rates and epoch settings.")
    clf = make_pipeline(Vectorizer(), StandardScaler(), SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42))
    clf.fit(np.asarray(X), np.asarray(y))
    joblib.dump(clf, model_file)


def authenticate(feature, expected_label: int, model_file: Path):
    clf = joblib.load(model_file)
    x = np.asarray(feature).reshape(1, *np.asarray(feature).shape)
    predicted = int(clf.predict(x)[0])
    probabilities = clf.predict_proba(x)[0]
    confidence = float(np.max(probabilities) * 100)
    return {
        "authenticated": predicted == int(expected_label),
        "predicted_label": predicted,
        "confidence": round(confidence, 1),
        "score_type": "SVM calibrated class probability (not a clinical or guaranteed identity score)",
    }


def profile_summary(username, meta, model_file):
    return {"username": username, "email": meta.get("email", ""), "label": meta.get("label"), "files": meta.get("files", 0), "model_ready": model_file.exists()}
