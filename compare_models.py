"""
Model Comparison Script
Compares all 3 trained models and recommends the best one.
"""

import os
import tensorflow as tf
from tensorflow.keras.models import load_model

MODELS_DIR = "./models"

models_info = [
    {
        "name": "best_modelllll.keras",
        "description": "Best val_accuracy checkpoint (saved by ModelCheckpoint during training)"
    },
    {
        "name": "model_66percent.keras",
        "description": "Snapshot at ~66% accuracy (backup before fine-tuning)"
    },
    {
        "name": "model_optionA_final.keras",
        "description": "Final model after continued training with lr=0.0001 (Option A fine-tune)"
    },
]

print("=" * 65)
print("         VIDEO ACTION RECOGNITION — MODEL COMPARISON")
print("=" * 65)

results = []

for m in models_info:
    path = os.path.join(MODELS_DIR, m["name"])
    size_mb = os.path.getsize(path) / 1e6

    print(f"\n[LOADING] {m['name']}")
    print(f"   Description : {m['description']}")
    print(f"   File size   : {size_mb:.2f} MB")

    try:
        model = load_model(path)
        total_params     = model.count_params()
        trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)

        print(f"   Total params     : {total_params:,}")
        print(f"   Trainable params : {trainable_params:,}")
        print(f"   Input shape      : {model.input_shape}")
        print(f"   Output shape     : {model.output_shape}")
        print(f"   Layers           : {len(model.layers)}")
        print(f"   ✅ Loaded successfully")

        results.append({
            "name": m["name"],
            "description": m["description"],
            "size_mb": size_mb,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "loaded": True
        })

    except Exception as e:
        print(f"   ❌ Failed to load: {e}")
        results.append({
            "name": m["name"],
            "loaded": False,
            "error": str(e)
        })

# ── SUMMARY TABLE ──────────────────────────────────────────────────
print("\n\n" + "=" * 65)
print("                     COMPARISON SUMMARY")
print("=" * 65)
print(f"{'Model':<30} {'Size (MB)':>10} {'Params':>15} {'Status':>10}")
print("-" * 65)

for r in results:
    if r["loaded"]:
        print(f"{r['name']:<30} {r['size_mb']:>10.2f} {r['total_params']:>15,}  ✅")
    else:
        print(f"{r['name']:<30} {'N/A':>10} {'N/A':>15}  ❌")

# ── RECOMMENDATION ─────────────────────────────────────────────────
print("\n" + "=" * 65)
print("                      RECOMMENDATION")
print("=" * 65)
print("""
  Model priority logic (based on notebook training flow):

  1. best_modelllll.keras
     → Saved by ModelCheckpoint monitoring val_accuracy
     → Always represents the HIGHEST validation accuracy
       achieved during the ENTIRE training run.
     → restore_best_weights=True in EarlyStopping also
       restores to this checkpoint.

  2. model_optionA_final.keras
     → Saved AFTER continued fine-tuning (Option A, lr=0.0001)
     → Could be better IF fine-tuning improved val_accuracy
       above the original best checkpoint.
     → But if fine-tuning overfit, this could be WORSE.

  3. model_66percent.keras
     → Explicitly named "66%" — this is a BACKUP snapshot
     → Lower accuracy than the others.
     → Only use as fallback.

  ⭐ RECOMMENDED: best_modelllll.keras
     Reason: ModelCheckpoint guarantees this is the epoch
     with the highest val_accuracy seen during training.
     It is the safest and most reliable choice.
     Use model_optionA_final.keras ONLY if you have test
     set metrics confirming it outperformed best_modelllll.
""")
print("=" * 65)
