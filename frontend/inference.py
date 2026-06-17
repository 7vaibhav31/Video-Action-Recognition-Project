"""
inference.py — Model loading and prediction pipeline.
Loads ResNet50 (feature extractor) and best_modelllll.keras (LSTM classifier)
once at startup. Exposes a predict(video_path) function.
"""

import os
import time
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

# ── Class Labels (same order as training) ─────────────────────────
SELECTED_CLASSES = [
    "Basketball", "TennisSwing", "GolfSwing", "Archery", "Bowling",
    "Diving", "Rafting", "Surfing",
    "PullUps", "PushUps", "Lunges", "JumpingJack",
    "Skiing", "HorseRiding", "Biking",
    "Typing", "Swing", "WalkingWithDog",
    "IceDancing", "PlayingGuitar",
]

# ── Config ─────────────────────────────────────────────────────────
NUM_FRAMES  = 16
IMG_SIZE    = 224
MODEL_PATH  = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_optionA_final.keras')

# ── Global model references (loaded once at startup) ───────────────
_cnn_model   = None
_lstm_model  = None


def _build_cnn():
    """Build frozen ResNet50 feature extractor."""
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    out = base(inp, training=False)
    out = GlobalAveragePooling2D()(out)
    return Model(inp, out, name='ResNet50_FeatureExtractor')


def load_models():
    """Load both models into memory. Call once at app startup."""
    global _cnn_model, _lstm_model
    print("[inference] Loading ResNet50 feature extractor...")
    _cnn_model = _build_cnn()
    print("[inference] ResNet50 loaded.")

    print(f"[inference] Loading LSTM classifier from: {MODEL_PATH}")
    _lstm_model = load_model(MODEL_PATH)
    print("[inference] LSTM model loaded. Ready for inference.")


def extract_frames(video_path):
    """
    Extract NUM_FRAMES uniformly sampled frames from a video.
    Returns numpy array of shape (NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3).
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return None

    frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255.0
            frames.append(frame)
        else:
            frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))

    cap.release()
    return np.array(frames, dtype=np.float32)


def predict(video_path):
    """
    Run full inference on a video file.

    Args:
        video_path: absolute path to the video file

    Returns:
        dict with keys:
            - predicted_class (str)
            - confidence (float, 0-100)
            - top3 (list of {label, confidence})
            - processing_time (float, seconds)
            - error (str, only if failed)
    """
    if _cnn_model is None or _lstm_model is None:
        return {"error": "Models not loaded. Call load_models() first."}

    start = time.time()

    try:
        # Step 1: Extract frames
        frames = extract_frames(video_path)
        if frames is None or len(frames) != NUM_FRAMES:
            return {"error": "Could not read video or video is too short."}

        # Step 2: Extract CNN features → (16, 2048)
        features = _cnn_model.predict(frames, verbose=0)  # (16, 2048)

        # Step 3: Add batch dim → (1, 16, 2048) and run LSTM
        features_batch = np.expand_dims(features, axis=0)
        probs = _lstm_model.predict(features_batch, verbose=0)[0]  # (20,)

        # Step 4: Build results
        top_idx = int(np.argmax(probs))
        top3_indices = np.argsort(probs)[::-1][:3]

        processing_time = round(time.time() - start, 2)

        return {
            "predicted_class": SELECTED_CLASSES[top_idx],
            "confidence": round(float(probs[top_idx]) * 100, 2),
            "top3": [
                {
                    "label": SELECTED_CLASSES[i],
                    "confidence": round(float(probs[i]) * 100, 2)
                }
                for i in top3_indices
            ],
            "processing_time": processing_time,
            "error": None
        }

    except Exception as e:
        return {"error": str(e)}
