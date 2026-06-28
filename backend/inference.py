"""
inference.py — Model loading and prediction pipeline.
Supports both Hugging Face Transformers and Custom CNN+LSTM Models.

CRITICAL: The custom .keras model was trained on Kaggle with Keras 3 (TF 2.21+),
but Hugging Face Spaces uses tensorflow-cpu==2.15 which ships with Keras 2.
Keras 3 saves new config keys (batch_shape, optional, quantization_config, etc.)
that Keras 2 does not understand, causing deserialization crashes.

This file monkey-patches the Keras 2 deserialization to strip those unknown keys
before they reach the layer constructors, making the load 100% compatible.
"""

import os
import time
import gc
import cv2
import numpy as np

# ── Global model references ───────────────
_video_pipeline = None
_custom_model = None
_cnn_extractor = None

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
# This section monkey-patches tf.keras so that .keras files saved by
# Keras 3 can be loaded by Keras 2 without crashing.
# ══════════════════════════════════════════════════════════════════════

def _apply_keras3_compat_patch():
    """
    Monkey-patch Keras 2's deserialization so that ALL layer types
    automatically strip Keras-3-only config keys before construction.
    
    Known Keras 3 keys that Keras 2 does NOT understand:
      - InputLayer: 'batch_shape' (Keras 2 uses 'batch_input_shape'), 'optional'
      - Dense: 'quantization_config'
      - Various: 'optional', 'quantization_config'
      
    Instead of patching each layer individually, we wrap the
    base_layer.from_config classmethod to clean configs universally.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import InputLayer, Dense
        from tensorflow.keras.engine import base_layer
        
        print("[inference][patch] Applying Keras 3 → Keras 2 compatibility patch...")

        # ── Patch 1: Fix InputLayer specifically ──────────────────
        # Keras 3 uses 'batch_shape', Keras 2 uses 'batch_input_shape'
        _original_input_layer_init = InputLayer.__init__

        def _patched_input_layer_init(self, **kwargs):
            # Convert Keras 3 'batch_shape' → Keras 2 'batch_input_shape'
            if 'batch_shape' in kwargs and 'batch_input_shape' not in kwargs:
                kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
                print(f"[inference][patch] InputLayer: converted batch_shape → batch_input_shape")
            elif 'batch_shape' in kwargs:
                kwargs.pop('batch_shape')
            
            # Remove 'optional' (Keras 3 only)
            if 'optional' in kwargs:
                kwargs.pop('optional')
                
            return _original_input_layer_init(self, **kwargs)

        InputLayer.__init__ = _patched_input_layer_init

        # ── Patch 2: Universal from_config cleaner ────────────────
        # This catches ANY layer type that receives unknown kwargs
        _original_from_config = base_layer.Layer.from_config

        @classmethod
        def _patched_from_config(cls, config):
            # List of keys that Keras 3 adds but Keras 2 doesn't recognize
            KERAS3_ONLY_KEYS = {'quantization_config', 'optional'}
            
            cleaned = False
            for key in list(config.keys()):
                if key in KERAS3_ONLY_KEYS:
                    config.pop(key)
                    cleaned = True
                    
            if cleaned:
                print(f"[inference][patch] {cls.__name__}: stripped Keras3-only keys from config")
                
            return _original_from_config.__func__(cls, config)

        base_layer.Layer.from_config = _patched_from_config
        
        print("[inference][patch] ✅ Keras compatibility patch applied successfully.")
        return True
        
    except Exception as e:
        print(f"[inference][patch] ⚠️ Could not apply Keras compat patch: {e}")
        print("[inference][patch] Will attempt model loading anyway...")
        return False


def load_models():
    """Load Hugging Face and Custom Keras models into memory. Call once at app startup."""
    global _video_pipeline, _custom_model, _cnn_extractor
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['OMP_NUM_THREADS'] = '1'  # Keep CPU usage stable
    
    # ── Step 1: Load Custom CNN+LSTM Model ────────────────────────
    # MUST be loaded BEFORE transformers to avoid the Keras 3→2 downgrade mid-load
    print("[inference] Loading Custom CNN+LSTM Model (kinetics_best_model.keras)...")
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.models import Model, load_model
        from tensorflow.keras.layers import GlobalAveragePooling2D, Input
        
        # Apply the Keras 3→2 compatibility patch BEFORE loading
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
                print(f"[inference] ✅ Custom CNN+LSTM model loaded successfully.")
            except Exception as e1:
                print(f"[inference] Direct load failed: {e1}")
                print("[inference] Attempting fallback: load with compile=False...")
                
                # Attempt 2: Load without compiling (skips optimizer state issues)
                try:
                    _custom_model = load_model(model_path, compile=False)
                    print(f"[inference] ✅ Custom model loaded with compile=False.")
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
        print("[inference] Custom model will not be available, but server will continue.")

    # ── Step 2: Load Hugging Face VideoMAE ────────────────────────
    print("[inference] Loading Hugging Face VideoMAE model...")
    try:
        from transformers import pipeline, VideoMAEImageProcessor
        processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        _video_pipeline = pipeline(
            "video-classification", 
            model="MCG-NJU/videomae-base-finetuned-kinetics", 
            image_processor=processor,
            device=-1 
        )
        print("[inference] ✅ VideoMAE loaded.")
    except ImportError as e:
        print(f"[inference] Error importing transformers: {e}")
        print("[inference] VideoMAE will not be available.")
    except Exception as e:
        print(f"[inference] ⚠️ Unexpected error loading VideoMAE: {e}")
        print("[inference] VideoMAE will not be available, but server will continue.")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n[inference] ══════ Model Loading Summary ══════")
    print(f"[inference]   Custom CNN+LSTM : {'✅ Ready' if _custom_model else '❌ Not loaded'}")
    print(f"[inference]   CNN Extractor   : {'✅ Ready' if _cnn_extractor else '❌ Not loaded'}")
    print(f"[inference]   HF VideoMAE     : {'✅ Ready' if _video_pipeline else '❌ Not loaded'}")
    print(f"[inference] ════════════════════════════════════\n")


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


def predict(video_path, model_type='transformer'):
    """
    Run full inference on a video file using the selected model.
    """
    global _video_pipeline, _custom_model
    
    # Lazy load models if they aren't loaded yet (e.g., Gunicorn worker edge cases)
    if _video_pipeline is None and _custom_model is None:
        print("[inference] No models in memory. Lazy loading now...")
        load_models()

    start = time.time()
    
    def format_label(label):
        # Format labels beautifully (e.g., 'playing basketball' -> 'Playing Basketball', 'SoccerJuggling' -> 'Soccer Juggling')
        import re
        s = label.replace('_', ' ')
        s = re.sub(r"([A-Z])", r" \1", s).strip()
        return " ".join([word.capitalize() for word in s.split()])

    try:
        if model_type == 'custom':
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
            # Transformer Model
            print(f"\n[inference] Running HF VideoMAE on {video_path}...")
            if _video_pipeline is None:
                return {"error": "Hugging Face model not loaded. Call load_models() first."}
                
            results = _video_pipeline(video_path, top_k=3)
            
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
        return {"error": str(e)}
    finally:
        gc.collect()
