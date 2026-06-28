# Deployment Memory & Troubleshooting Log
**Project:** ActionNet (Video Action Recognition)
**Deployment Targets:** Hugging Face Spaces (Backend/Docker) & Vercel (Frontend)

This file serves as a memory bank of all the critical edge-cases, bugs, and fixes we applied during the Hugging Face Docker deployment. If the server crashes in the future, check this file first.

---

## 1. The Gunicorn Startup Bug (Lazy Loading Fix)
* **The Problem**: Hugging Face's default Docker builder runs `gunicorn` with its own command line arguments, completely ignoring our `gunicorn.conf.py` startup hooks. This meant the models were never loaded into memory on boot, resulting in a `ValueError: Hugging Face model not loaded. Call load_models() first.`
* **The Fix**: We implemented **Lazy Loading** inside `inference.py`. The `predict()` function now explicitly checks `if _video_pipeline is None or _custom_model is None:` before running. If they are missing, it triggers `load_models()` on the fly during the very first video upload.

## 2. The Hugging Face `transformers` Dependency Crash
* **The Problem**: When the Hugging Face Space booted up the `VideoMAE` model, it threw an error: `VideoMAEImageProcessor requires the Torchvision library... requires the PIL library`.
* **The Fix**: We explicitly added `torchvision>=0.15.0` and `pillow>=9.0.0` to the `requirements.txt` file.

## 3. The Keras 3 vs Keras 2 Deserialization Conflict (CRITICAL)
* **The Problem**: The custom CNN+LSTM model was trained using a newer version of TensorFlow (Keras 3). However, when the Hugging Face `transformers` library loads, it forces TensorFlow to downgrade to Keras 2 (`tf-keras`) for compatibility. When Keras 2 tried to read the Keras 3 model, it crashed on an unrecognized keyword argument: `Unrecognized keyword arguments passed to Dense: {'quantization_config': None}`.
* **The Fix**: 
  1. **Loading Order**: We swapped the loading order in `inference.py` so that TensorFlow and the custom `.keras` model are loaded *first*, before `transformers` is even imported. This prevents the secret downgrade from happening during deserialization.
  2. **Custom Objects (SafeDense)**: We created a custom object wrapper (`SafeDense`) inside `load_model(..., custom_objects={'Dense': SafeDense})` that explicitly intercepts and deletes the `quantization_config` parameter right before it loads into memory.

## 4. Model Path Fallback
* **The Problem**: The user dragged the `kinetics_best_model.keras` file directly into the root folder of the Hugging Face space instead of placing it inside a `models/` directory. The code crashed trying to find the path.
* **The Fix**: We added a fallback `if/else` path check in `inference.py`. It looks in `models/` first. If it doesn't exist, it safely checks the root directory (`__file__`).

---

## What to do if another error occurs?
1. Go to the Hugging Face Spaces **Logs** tab.
2. Read the very bottom of the log output for a `Traceback` or `Exception`.
3. If it's a `ModuleNotFoundError`, we need to add the missing library to `requirements.txt`.
4. If it's an Out-of-Memory error, we may need to shrink the video extraction frames in `compress.py` or restrict the `max_length` in `inference.py`.
5. Every time you fix code, remember to upload the newly updated file to the Hugging Face **Files** tab and wait for the **Running** status.

---

## 5. The InputLayer `batch_shape` / `optional` Crash (CRITICAL — 2026-06-28)
* **The Problem**: The model was trained on Kaggle with Keras 3 (TF 2.21+), but the Docker container uses `tensorflow-cpu==2.15.0` which ships with Keras 2. Keras 3 saves `InputLayer` config with `batch_shape` and `optional` keys. Keras 2 only understands `batch_input_shape` and crashes with: `Unrecognized keyword arguments: ['batch_shape', 'optional']`.
* **The Fix**: 
  1. **Monkey-patch `InputLayer.__init__`**: We intercept the constructor and rename `batch_shape` → `batch_input_shape`, and strip `optional`.
  2. **Universal `from_config` cleaner**: We patch `base_layer.Layer.from_config` to strip ALL Keras-3-only keys (`quantization_config`, `optional`) from EVERY layer type's config before construction.
  3. This replaces the old per-layer `SafeDense` approach with a single universal fix.

## 6. The Gunicorn Infinite Boot Crash Loop Fix (2026-06-28)
* **The Problem**: `load_models()` was called directly at module-level in `app.py` (line 165). If model loading threw ANY unhandled exception, the gunicorn worker would crash, restart, crash again — in an infinite loop. HF Spaces would show "Building" forever.
* **The Fix**: Wrapped the `load_models()` call in `try/except` in both the `__main__` and `else` (gunicorn) branches. Now the server ALWAYS boots. If models fail to load at startup, they will be lazy-loaded on the first prediction request.

## 7. Wrong Import Path for Keras Patch (2026-06-28)
* **The Problem**: The Keras compat patch tried `from tensorflow.keras.engine import base_layer` which fails on TF 2.15 because `tensorflow.keras.engine` is NOT exposed as a submodule. Error: `No module named 'tensorflow.keras.engine'`.
* **The Fix**: Import from the standalone `keras` package instead, matching the actual installed paths shown in the tracebacks:
  1. `from keras.src.engine.input_layer import InputLayer` (for InputLayer patch)
  2. `from keras.src.engine.base_layer import Layer` (for universal from_config patch)
  3. Falls back to `tensorflow.keras.layers` (public API) if internal paths change.

## 8. The torchvision `read_video` Crash (2026-06-28)
* **The Problem**: The HF Transformers `pipeline("video-classification")` internally calls `torchvision.io.read_video()` to read video files. But the installed torchvision version removed this function: `module 'torchvision.io' has no attribute 'read_video'`.
* **The Fix**: **Bypassed the pipeline entirely.** Instead of using `pipeline(...)`, we now:
  1. Load `VideoMAEImageProcessor` and `VideoMAEForVideoClassification` separately.
  2. Read video frames ourselves using **OpenCV** (`_read_video_opencv()`).
  3. Process frames through the image processor manually.
  4. Run the model directly with `torch.no_grad()`.
  5. This eliminates ALL dependency on `torchvision.io` for video reading.

