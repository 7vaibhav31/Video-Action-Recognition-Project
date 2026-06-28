"""
inference.py — Model loading and prediction pipeline.
Supports both Hugging Face VideoMAE Transformer and Custom CNN+LSTM Models.

FIX HISTORY:
  - Keras 3→2 compat: Model saved with Keras 3 (batch_shape, optional, quantization_config)
    but HF Spaces runs tensorflow-cpu==2.15 (Keras 2). We monkey-patch the actual Keras
    classes using 'keras.src.engine.*' imports (matching the installed package paths).
  - VideoMAE torchvision fix: The pipeline crashes because torchvision.io.read_video is
    missing in newer torchvision. We bypass the pipeline entirely — read frames with OpenCV,
    process with VideoMAEImageProcessor, and run the model directly.
"""

import os
import time
import gc
import cv2
import numpy as np

# ── Global model references ───────────────
_custom_model = None
_cnn_extractor = None
_video_processor = None   # VideoMAEImageProcessor
_video_model = None        # VideoMAEForVideoClassification

# UCF50 Classes (50)
UCF50_CLASSES = [
    'BaseballPitch', 'Basketball', 'BenchPress', 'Biking', 'Billiards', 
    'BreastStroke', 'CleanAndJerk', 'Diving', 'Drumming', 'Fencing', 
    'GolfSwing', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop', 
    'JavelinThrow', 'JugglingBalls', 'JumpRope', 'JumpingJack', 'Kayaking', 
    'Lunges', 'MilitaryParade', 'Mixing', 'Nunchucks', 'PizzaTossing', 
    'PlayingGuitar', 'PlayingPiano', 'PlayingTabla', 'PlayingViolin', 'PoleVault', 
    'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'RockClimbingIndoor', 
    'RopeClimbing', 'Rowing', 'SalsaSpin', 'SkateBoarding', 'Skiing', 
    'Skijet', 'SoccerJuggling', 'Swing', 'TaiChi', 'TennisSwing', 
    'ThrowDiscus', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog', 'YoYo'
]


# ══════════════════════════════════════════════════════════════════════
# KERAS 3 → KERAS 2 COMPATIBILITY PATCH
# ══════════════════════════════════════════════════════════════════════

def _apply_keras3_compat_patch():
    """
    Monkey-patch Keras 2 layer constructors so that .keras model files
    saved by Keras 3 can load without crashing.

    Keras 3 config keys that Keras 2 doesn't understand:
      - InputLayer: 'batch_shape' (K2 uses 'batch_input_shape'), 'optional'
      - Dense/LSTM/etc: 'quantization_config'
    """
    try:
        import tensorflow as tf
        print("[inference][patch] Applying Keras 3 → Keras 2 compatibility patch...")

        # ── 1. Patch InputLayer.__init__ ──────────────────────────
        # Get the ACTUAL InputLayer class used during deserialization.
        # Try the standalone keras package first (matches HF Spaces traceback paths),
        # then fall back to tensorflow.keras (public API, same class object).
        InputLayer = None
        for import_path in [
            ('keras.src.engine.input_layer', 'InputLayer'),
            ('keras.engine.input_layer', 'InputLayer'),
        ]:
            try:
                mod = __import__(import_path[0], fromlist=[import_path[1]])
                InputLayer = getattr(mod, import_path[1])
                print(f"[inference][patch] Found InputLayer via {import_path[0]}")
                break
            except (ImportError, AttributeError):
                continue

        if InputLayer is None:
            # Fall back to the public TF API (should be the same class)
            InputLayer = tf.keras.layers.InputLayer
            print("[inference][patch] Found InputLayer via tf.keras.layers")

        _orig_input_init = InputLayer.__init__

        def _patched_input_init(self, **kwargs):
            # Convert Keras 3 'batch_shape' → Keras 2 'batch_input_shape'
            if 'batch_shape' in kwargs:
                if 'batch_input_shape' not in kwargs:
                    kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
                else:
                    kwargs.pop('batch_shape')
            # Strip Keras 3 only keys
            kwargs.pop('optional', None)
            kwargs.pop('quantization_config', None)
            return _orig_input_init(self, **kwargs)

        InputLayer.__init__ = _patched_input_init
        print("[inference][patch] ✅ InputLayer.__init__ patched")

        # ── 2. Patch base Layer.from_config (universal) ───────────
        # This catches ALL layer types (Dense, LSTM, Dropout, etc.)
        BaseLayer = None
        for import_path in [
            ('keras.src.engine.base_layer', 'Layer'),
            ('keras.engine.base_layer', 'Layer'),
        ]:
            try:
                mod = __import__(import_path[0], fromlist=[import_path[1]])
                BaseLayer = getattr(mod, import_path[1])
                print(f"[inference][patch] Found base Layer via {import_path[0]}")
                break
            except (ImportError, AttributeError):
                continue

        if BaseLayer is None:
            BaseLayer = tf.keras.layers.Layer
            print("[inference][patch] Found base Layer via tf.keras.layers")

        _orig_from_config = BaseLayer.from_config

        @classmethod
        def _patched_from_config(cls, config):
            # Strip Keras 3 only keys from ANY layer's config
            for key in ['quantization_config', 'optional']:
                config.pop(key, None)
            return _orig_from_config.__func__(cls, config)

        BaseLayer.from_config = _patched_from_config
        print("[inference][patch] ✅ Layer.from_config patched (universal)")

        print("[inference][patch] ✅ All Keras compatibility patches applied successfully.")
        return True

    except Exception as e:
        print(f"[inference][patch] ⚠️ Could not apply Keras compat patch: {e}")
        import traceback
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════
# VIDEO FRAME READING (OpenCV — avoids torchvision dependency)
# ══════════════════════════════════════════════════════════════════════

def _read_video_opencv(video_path, num_frames=16):
    """
    Read exactly `num_frames` uniformly-sampled frames from a video using OpenCV.
    Returns a list of numpy arrays (H, W, 3) in RGB uint8 format.
    This completely bypasses torchvision.io.read_video.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        print(f"[inference] Warning: Could not read frame count from {video_path}")
        return None

    # Sample num_frames uniformly across the video
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # OpenCV reads BGR, convert to RGB for the model
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            # If frame read fails, duplicate last good frame
            if frames:
                frames.append(frames[-1].copy())

    cap.release()

    # Ensure we have exactly num_frames (pad with last frame if short)
    while len(frames) < num_frames:
        if frames:
            frames.append(frames[-1].copy())
        else:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

    return frames[:num_frames]


# ══════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════

def load_models():
    """Load Hugging Face and Custom Keras models into memory. Call once at app startup."""
    global _custom_model, _cnn_extractor, _video_processor, _video_model

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['OMP_NUM_THREADS'] = '1'

    # ── Step 1: Load Custom CNN+LSTM Model ────────────────────────
    # MUST be loaded BEFORE transformers to avoid Keras version conflicts mid-load
    print("[inference] Loading Custom CNN+LSTM Model (kinetics_best_model.keras)...")
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.models import Model, load_model
        from tensorflow.keras.layers import GlobalAveragePooling2D, Input

        # Apply the Keras 3 → Keras 2 compatibility patch BEFORE loading
        _apply_keras3_compat_patch()

        # Find the model file (check 'models/' first, then root)
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'kinetics_best_model.keras')
        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(__file__), 'kinetics_best_model.keras')

        if os.path.exists(model_path):
            print(f"[inference] Found model at: {model_path}")

            # Attempt 1: Direct load with patched Keras
            try:
                _custom_model = load_model(model_path)
                print("[inference] ✅ Custom CNN+LSTM model loaded successfully.")
            except Exception as e1:
                print(f"[inference] Direct load failed: {e1}")
                print("[inference] Attempting fallback: load with compile=False...")

                # Attempt 2: Load without compiling (skips optimizer state issues)
                try:
                    _custom_model = load_model(model_path, compile=False)
                    print("[inference] ✅ Custom model loaded with compile=False.")
                except Exception as e2:
                    print(f"[inference] ❌ All load attempts failed: {e2}")
                    _custom_model = None

            # Setup ResNet50 feature extractor (only if model loaded)
            if _custom_model is not None:
                base_cnn = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
                cnn_input = Input(shape=(224, 224, 3))
                cnn_out = base_cnn(cnn_input, training=False)
                cnn_out = GlobalAveragePooling2D()(cnn_out)
                _cnn_extractor = Model(cnn_input, cnn_out, name='CNN_FeatureExtractor')
                print("[inference] ✅ ResNet50 CNN Extractor loaded.")
        else:
            print(f"[inference] ⚠️ Custom model file not found. Skipping custom model.")

    except ImportError as e:
        print(f"[inference] Error importing tensorflow: {e}")
        print("[inference] Custom model will not be available.")
    except Exception as e:
        print(f"[inference] ⚠️ Unexpected error loading custom model: {e}")
        import traceback
        traceback.print_exc()

    # ── Step 2: Load Hugging Face VideoMAE ────────────────────────
    # We load the processor and model SEPARATELY (not as a pipeline)
    # so we can feed OpenCV-read frames directly, bypassing torchvision.io.read_video
    print("[inference] Loading Hugging Face VideoMAE model...")
    try:
        from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

        _video_processor = VideoMAEImageProcessor.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics"
        )
        _video_model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics"
        )
        _video_model.eval()  # Set to evaluation mode
        print("[inference] ✅ VideoMAE model + processor loaded (direct mode, no torchvision needed).")

    except ImportError as e:
        print(f"[inference] Error importing transformers: {e}")
        print("[inference] VideoMAE will not be available.")
    except Exception as e:
        print(f"[inference] ⚠️ Unexpected error loading VideoMAE: {e}")
        import traceback
        traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n[inference] ══════ Model Loading Summary ══════")
    print(f"[inference]   Custom CNN+LSTM : {'✅ Ready' if _custom_model else '❌ Not loaded'}")
    print(f"[inference]   CNN Extractor   : {'✅ Ready' if _cnn_extractor else '❌ Not loaded'}")
    print(f"[inference]   HF VideoMAE     : {'✅ Ready' if _video_model else '❌ Not loaded'}")
    print(f"[inference]   VideoMAE Proc.  : {'✅ Ready' if _video_processor else '❌ Not loaded'}")
    print(f"[inference] ════════════════════════════════════\n")


# ══════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (for Custom CNN+LSTM model)
# ══════════════════════════════════════════════════════════════════════

def extract_features(video_path, num_frames=30, img_size=224):
    """Extract ResNet50 features from video for the custom model."""
    from tensorflow.keras.applications.resnet50 import preprocess_input

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return None

    # Focus on the middle of the video (max 150 frames)
    MAX_ACTION_FRAMES = 150
    if total_frames <= MAX_ACTION_FRAMES:
        start_frame = 0
        end_frame = total_frames - 1
    else:
        start_frame = (total_frames - MAX_ACTION_FRAMES) // 2
        end_frame = start_frame + MAX_ACTION_FRAMES - 1

    frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Center crop
            h, w = frame.shape[:2]
            min_dim = min(h, w)
            start_x = (w - min_dim) // 2
            start_y = (h - min_dim) // 2
            frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

            # Resize and color space
            frame = cv2.resize(frame, (img_size, img_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            frames.append(np.zeros((img_size, img_size, 3)))

    cap.release()
    frames = np.array(frames, dtype=np.float32)
    frames = preprocess_input(frames)

    # Extract CNN Features
    features = _cnn_extractor.predict(frames, verbose=0)
    return features


# ══════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════

def predict(video_path, model_type='transformer'):
    """
    Run full inference on a video file using the selected model.
    """
    global _custom_model, _video_processor, _video_model

    # Lazy load models if none are loaded yet
    if _video_model is None and _custom_model is None:
        print("[inference] No models in memory. Lazy loading now...")
        load_models()

    start = time.time()

    def format_label(label):
        """Format labels beautifully (e.g., 'SoccerJuggling' -> 'Soccer Juggling')."""
        import re
        s = label.replace('_', ' ')
        s = re.sub(r"([A-Z])", r" \1", s).strip()
        return " ".join([word.capitalize() for word in s.split()])

    try:
        if model_type == 'custom':
            # ── Custom CNN+LSTM Inference ─────────────────────────
            print(f"\n[inference] Running Custom CNN+LSTM on {video_path}...")
            if _custom_model is None or _cnn_extractor is None:
                return {"error": "Custom model not loaded properly. Check logs."}

            features = extract_features(video_path, num_frames=30)
            if features is None or len(features) != 30:
                return {"error": "Failed to extract features from video."}

            # Model expects batch dimension: (1, 30, 2048)
            features_batch = np.expand_dims(features, axis=0)

            # Predict
            predictions = _custom_model.predict(features_batch, verbose=0)[0]

            # Get top 3
            top_indices = np.argsort(predictions)[-3:][::-1]
            results = []
            for idx in top_indices:
                results.append({
                    "label": UCF50_CLASSES[idx] if idx < len(UCF50_CLASSES) else f"Class_{idx}",
                    "score": float(predictions[idx])
                })

        else:
            # ── VideoMAE Transformer Inference ────────────────────
            # We use the processor + model directly (NOT the pipeline)
            # so we can read frames with OpenCV, bypassing torchvision
            print(f"\n[inference] Running HF VideoMAE on {video_path}...")
            if _video_model is None or _video_processor is None:
                return {"error": "Hugging Face model not loaded. Check logs."}

            import torch

            # 1. Read 16 frames from video using OpenCV (no torchvision needed!)
            frames = _read_video_opencv(video_path, num_frames=16)
            if frames is None or len(frames) == 0:
                return {"error": "Failed to read video frames."}

            print(f"[inference] Read {len(frames)} frames via OpenCV (shape: {frames[0].shape})")

            # 2. Process frames with VideoMAE processor
            inputs = _video_processor(frames, return_tensors="pt")

            # 3. Run the model (no gradient needed for inference)
            with torch.no_grad():
                outputs = _video_model(**inputs)

            # 4. Get top-3 predictions
            logits = outputs.logits[0]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            top3_values, top3_indices = torch.topk(probs, 3)

            results = []
            for score, idx in zip(top3_values, top3_indices):
                label = _video_model.config.id2label[idx.item()]
                results.append({
                    "label": label,
                    "score": score.item()
                })

        print(f"[inference] Inference complete. Raw results: {results}")

        if not results:
            return {"error": "Model returned empty predictions."}

        top_label = results[0]['label']
        top_score = results[0]['score']
        processing_time = round(time.time() - start, 2)

        return {
            "predicted_class": format_label(top_label),
            "confidence": round(float(top_score) * 100, 2),
            "top3": [
                {
                    "label": format_label(res['label']),
                    "confidence": round(float(res['score']) * 100, 2)
                }
                for res in results
            ],
            "processing_time": processing_time,
            "error": None
        }

    except Exception as e:
        print(f"[inference] EXCEPTION encountered during prediction: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    finally:
        gc.collect()
