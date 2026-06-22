"""
inference.py — Model loading and prediction pipeline.
Loads ResNet50 (feature extractor) and best_modelllll.keras (LSTM classifier)
once at startup. Exposes a predict(video_path) function.
"""

import os

# Limit TensorFlow thread-pool allocations before importing to conserve memory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import time
import gc
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
# ── Model Path (Supports local and Vercel serverless) ───────────────
_local_path = os.path.join(os.path.dirname(__file__), 'model_optionA_final.keras')
if os.path.exists(_local_path):
    MODEL_PATH = _local_path
else:
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'model_optionA_final.keras')

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
    
    # Optimize TensorFlow CPU performance on limited hosting environments (like Render Free CPU)
    # Restricts thread spawning overhead to prevent CPU throttling and context switching
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        print("[inference] TensorFlow CPU threading optimized (threads set to 1).")
    except Exception as e:
        print(f"[inference] Warning: Could not set TensorFlow threading configuration: {e}")

    print("[inference] Loading ResNet50 feature extractor...")
    _cnn_model = _build_cnn()
    print("[inference] ResNet50 loaded.")

    print(f"[inference] Loading LSTM classifier from: {MODEL_PATH}")
    _lstm_model = load_model(MODEL_PATH)
    print("[inference] LSTM model loaded. Ready for inference.")


def extract_frames_opencv(video_path):
    """Attempt to extract frames using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[inference] OpenCV: VideoCapture failed to open file.")
        cap.release()
        return None
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[inference] OpenCV: Total frames in video: {total_frames}")

    if total_frames <= 0:
        print("[inference] OpenCV: Total frames <= 0. Attempting sequential read...")
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
        total_frames = count
        print(f"[inference] OpenCV: Counted {total_frames} frames sequentially.")
        cap.release()
        cap = cv2.VideoCapture(video_path)
        if total_frames <= 0:
            cap.release()
            return None

    frame_indices = set(np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int))
    print(f"[inference] OpenCV: Sampling 16 frames at indices: {list(sorted(frame_indices))}")
    
    frames = []
    count = 0
    while len(frames) < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if count in frame_indices:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255.0
            frames.append(frame)
        count += 1
        
    cap.release()

    # Only pad if we got at least 1 frame, otherwise return None to fallback
    if len(frames) > 0:
        while len(frames) < NUM_FRAMES:
            print(f"[inference] OpenCV Warning: padded frame with zeros (got {len(frames)} of {NUM_FRAMES})")
            frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))
        return np.array(frames, dtype=np.float32)
    return None


def extract_frames_pyav(video_path):
    """Attempt to extract frames using PyAV (av)."""
    import av
    container = av.open(video_path)
    try:
        if not container.streams.video:
            print("[inference] PyAV: No video stream found in container.")
            return None
        stream = container.streams.video[0]
        total_frames = stream.frames
        
        if total_frames is None or total_frames <= 0:
            print("[inference] PyAV: Stream frames <= 0 or None. Counting sequentially...")
            total_frames = 0
            for frame in container.decode(video=0):
                total_frames += 1
            print(f"[inference] PyAV: Counted {total_frames} frames sequentially.")
            container.close()
            container = av.open(video_path)
            
        if total_frames <= 0:
            print("[inference] PyAV: Total counted frames <= 0.")
            return None
            
        frame_indices = set(np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int))
        print(f"[inference] PyAV: Sampling 16 frames at indices: {list(sorted(frame_indices))}")
        
        frames = []
        count = 0
        for frame in container.decode(video=0):
            if count in frame_indices:
                img = frame.to_ndarray(format='rgb24')
                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img_normalized = img_resized / 255.0
                frames.append(img_normalized)
                if len(frames) == NUM_FRAMES:
                    break
            count += 1
            
        if len(frames) > 0:
            while len(frames) < NUM_FRAMES:
                print(f"[inference] PyAV Warning: padded frame with zeros (got {len(frames)} of {NUM_FRAMES})")
                frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))
            return np.array(frames, dtype=np.float32)
        return None
    finally:
        container.close()


def extract_frames(video_path):
    """
    Extract NUM_FRAMES uniformly sampled frames from a video.
    Tries OpenCV first, and falls back to PyAV if OpenCV fails.
    Returns tuple (numpy array, error_message).
    """
    print(f"\n[inference] Loading video file: {video_path}")
    
    cv_error = None
    pyav_error = None
    
    # 1. Try OpenCV
    try:
        frames = extract_frames_opencv(video_path)
        if frames is not None and len(frames) == NUM_FRAMES:
            print("[inference] OpenCV extraction succeeded.")
            return frames, None
        cv_error = "OpenCV returned 0 or incomplete frames."
    except Exception as e:
        cv_error = f"OpenCV Exception: {str(e)}"
        print(f"[inference] OpenCV extraction failed with exception: {e}")
        
    print("[inference] OpenCV failed. Trying PyAV as fallback...")
    
    # 2. Try PyAV
    try:
        frames = extract_frames_pyav(video_path)
        if frames is not None and len(frames) == NUM_FRAMES:
            print("[inference] PyAV extraction succeeded.")
            return frames, None
        pyav_error = "PyAV returned 0 or incomplete frames."
    except Exception as e:
        pyav_error = f"PyAV Exception: {str(e)}"
        print(f"[inference] PyAV extraction failed with exception: {e}")
        
    combined_error = f"OpenCV Error: {cv_error} | PyAV Error: {pyav_error}"
    print(f"[inference] ERROR: All frame extraction backends failed. {combined_error}")
    return None, combined_error


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
        frames, err_msg = extract_frames(video_path)
        if frames is None:
            return {"error": f"Could not read video: {err_msg if err_msg else 'Unknown frame extraction error.'}"}

        # Step 2: Extract CNN features → (16, 2048)
        print("[inference] Running ResNet50 feature extraction...")
        features = _cnn_model.predict(frames, verbose=0)  # (16, 2048)
        print(f"[inference] Feature extraction complete. Output shape: {features.shape}")

        # Step 3: Add batch dim → (1, 16, 2048) and run LSTM
        print("[inference] Running LSTM classification...")
        features_batch = np.expand_dims(features, axis=0)
        probs = _lstm_model.predict(features_batch, verbose=0)[0]  # (20,)
        
        # Step 4: Build results
        top_idx = int(np.argmax(probs))
        top3_indices = np.argsort(probs)[::-1][:3]
        
        print(f"[inference] Model prediction complete. Top Class: {SELECTED_CLASSES[top_idx]} ({probs[top_idx]*100:.2f}%)")
        print(f"[inference] Top 3 probabilities: " + ", ".join([f"{SELECTED_CLASSES[i]}: {probs[i]*100:.1f}%" for i in top3_indices]))

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
        print(f"[inference] EXCEPTION encountered during prediction: {e}")
        return {"error": str(e)}
    finally:
        # Explicitly run garbage collection to free temporary tensors from memory
        gc.collect()
