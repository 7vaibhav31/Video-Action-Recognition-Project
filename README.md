<div align="center">

<!-- Animated Header -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:000000,100:1a1a1a&height=200&section=header&text=ActionNet&fontSize=80&fontColor=ffffff&fontAlignY=38&desc=Video%20Action%20Recognition%20%7C%20ResNet50%20%2B%20LSTM&descAlignY=58&descSize=18" width="100%"/>

<br/>

<!-- Badges -->
![Python](https://img.shields.io/badge/Python-3.12-ffffff?style=for-the-badge&logo=python&logoColor=black&labelColor=000000)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-ffffff?style=for-the-badge&logo=tensorflow&logoColor=black&labelColor=000000)
![Keras](https://img.shields.io/badge/Keras-3.14-ffffff?style=for-the-badge&logo=keras&logoColor=black&labelColor=000000)
![Flask](https://img.shields.io/badge/Flask-3.1-ffffff?style=for-the-badge&logo=flask&logoColor=black&labelColor=000000)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-ffffff?style=for-the-badge&logo=opencv&logoColor=black&labelColor=000000)
![Vercel](https://img.shields.io/badge/Vercel-Ready-ffffff?style=for-the-badge&logo=vercel&logoColor=black&labelColor=000000)

<br/>

> **A deep learning system that watches a video and understands what humans are doing.**
> Built with a ResNet50 + LSTM pipeline, served via a stunning black & white Flask dashboard.

<br/>

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Try%20It%20Now-white?style=for-the-badge&labelColor=000)](https://github.com/7vaibhav31/Video-Action-Recognition-Project)
[![Dataset](https://img.shields.io/badge/📦%20Dataset-UCF--101-white?style=for-the-badge&labelColor=000)](https://www.kaggle.com/datasets/pevogam/ucf101/data)

</div>

---

## ✦ What Is This?

**ActionNet** is a full-stack video action recognition system. You drop in a video — it tells you what action is happening, with confidence scores and a top-3 prediction breakdown.

Under the hood, every video is converted into 16 uniformly-sampled frames. Each frame is passed through a frozen **ResNet50** (pretrained on ImageNet) to extract 2,048 spatial features. These 16 feature vectors form a sequence, which a **2-layer LSTM** reads to understand the temporal pattern — and finally classifies into one of **20 human action classes**.

All of this runs inside a professional **Flask web app** with a fully animated black & white dashboard.

---

## ✦ Live Dashboard Preview

```
┌─────────────────────────────────────────────────────────┐
│  ◉ ActionNet                    Dashboard  Model  Docs  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│         VIDEO ACTION RECOGNITION                        │
│         AI-powered. Real-time. Precise.                 │
│                                                         │
│  ┌───────────────────┐  ┌──────────────────────────┐   │
│  │  ╔═══════════════╗│  │  Predicted Action        │   │
│  │  ║  Drop Video  ║│  │  ┌────────────────────┐  │   │
│  │  ║   or click   ║│  │  │   Basketball  🏀   │  │   │
│  │  ╚═══════════════╝│  │  │   ████████░░  82%  │  │   │
│  │                   │  │  └────────────────────┘  │   │
│  │  [Analyze Video]  │  │  Top 3 ───────────────── │   │
│  └───────────────────┘  │  #1 Basketball     82.1% │   │
│                         │  #2 Biking          9.3% │   │
│                         │  #3 HorseRiding     4.2% │   │
│                         └──────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## ✦ Architecture

```
Raw Video (.mp4 / .avi / .mov)
         │
         ▼
  ┌─────────────────┐
  │  OpenCV         │  Uniform sampling → exactly 16 frames
  │  Frame Sampler  │  Each resized to 224 × 224 × 3
  └────────┬────────┘
           │  (16, 224, 224, 3)
           ▼
  ┌─────────────────┐
  │  ResNet50       │  Pretrained on ImageNet
  │  (Frozen CNN)   │  Global Average Pooling
  └────────┬────────┘
           │  (16, 2048)  — spatial features per frame
           ▼
  ┌─────────────────┐
  │  LSTM Layer 1   │  256 units  — return_sequences=True
  │  Dropout 0.3    │
  ├─────────────────┤
  │  LSTM Layer 2   │  128 units  — return_sequences=False
  │  Dropout 0.3    │
  │  BatchNorm      │
  ├─────────────────┤
  │  Dense 64 ReLU  │
  │  Dropout 0.2    │
  ├─────────────────┤
  │  Dense 20       │  Softmax — one per action class
  └────────┬────────┘
           │
           ▼
  { predicted_class, confidence, top3 }
```

---

## ✦ 20 Supported Action Classes

| Category | Actions |
|---|---|
| 🏀 **Sports** | Basketball, TennisSwing, GolfSwing, Archery, Bowling |
| 🌊 **Water / Outdoor** | Diving, Rafting, Surfing, Skiing, HorseRiding, Biking |
| 💪 **Fitness** | PullUps, PushUps, Lunges, JumpingJack |
| 🎵 **Daily Life & Arts** | Typing, Swing, WalkingWithDog, IceDancing, PlayingGuitar |

---

## ✦ Project Structure

```
Video-Action-Recognition-Project/
│
├── 📓 Notebook/
│   └── video-recognition-project.ipynb   ← Full training pipeline
│
├── 🌐 frontend/
│   ├── app.py                             ← Flask application
│   ├── inference.py                       ← Prediction pipeline
│   ├── requirements.txt                   ← Frontend dependencies
│   ├── vercel.json                        ← Vercel deployment config
│   ├── templates/
│   │   └── index.html                     ← Animated dashboard
│   └── static/
│       ├── css/style.css                  ← Black & white design system
│       └── js/main.js                     ← Canvas animation + upload
│
├── 🤖 models/                             ← Trained model files (download separately)
│   ├── best_modelllll.keras               ← Best val_accuracy checkpoint
│   ├── model_optionA_final.keras          ← Fine-tuned final model
│   └── model_66percent.keras              ← Backup snapshot
│
├── requirements.txt                       ← Full project dependencies
├── compare_models.py                      ← Model evaluation script
└── .gitignore
```

---

## ✦ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/7vaibhav31/Video-Action-Recognition-Project.git
cd Video-Action-Recognition-Project
```

### 2. Create & Activate Virtual Environment
```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the Trained Model
Download `model_optionA_final.keras` from the [Releases](https://github.com/7vaibhav31/Video-Action-Recognition-Project/releases) and place it in the `models/` directory.

### 5. Run the Web App
```bash
cd frontend
python app.py
```

Open **[http://localhost:5000](http://localhost:5000)** in your browser.

---

## ✦ Training Pipeline

The full training notebook is in `Notebook/video-recognition-project.ipynb`.

**Dataset:** [UCF-101](https://www.kaggle.com/datasets/pevogam/ucf101/data) — 20 selected action classes

| Stage | Details |
|---|---|
| Frame Extraction | 16 uniform frames per video via OpenCV |
| CNN Feature Extraction | ResNet50 (ImageNet weights, last 20 layers unfrozen) |
| Feature Storage | `(N, 16, 2048)` arrays saved as `.npy` |
| Data Split | 80% train / 10% val / 10% test (stratified) |
| LSTM Training | Adam lr=0.0001, EarlyStopping patience=7, ReduceLROnPlateau |
| Best Checkpoint | Saved by ModelCheckpoint monitoring `val_accuracy` |

```python
# Training callbacks used
ModelCheckpoint(monitor='val_accuracy', save_best_only=True)
EarlyStopping(patience=7, restore_best_weights=True)
ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6)
```

---

## ✦ Model Comparison

| Model | Description | Recommended |
|---|---|---|
| `best_modelllll.keras` | Highest val_accuracy checkpoint | ✅ Safe choice |
| `model_optionA_final.keras` | Fine-tuned with lr=0.0001 | ✅ Better generalization |
| `model_66percent.keras` | 66% accuracy backup snapshot | ⚠️ Backup only |

Run the comparison script:
```bash
python compare_models.py
```

---

## ✦ API Reference

### `POST /predict`

Upload a video and receive an action prediction.

**Request:**
```
Content-Type: multipart/form-data
Body: video=<file>  (MP4, AVI, MOV, MKV, WebM — max 200MB)
```

**Response:**
```json
{
  "predicted_class": "Basketball",
  "confidence": 82.14,
  "top3": [
    { "label": "Basketball",  "confidence": 82.14 },
    { "label": "Biking",      "confidence":  9.32 },
    { "label": "HorseRiding", "confidence":  4.21 }
  ],
  "processing_time": 8.3,
  "error": null
}
```

### `GET /health`
```json
{ "status": "ok", "message": "Video Action Recognition API is running." }
```

---

## ✦ Deployment

### Vercel (Frontend)
```bash
cd frontend
vercel --prod
```

### Render / Railway (Backend — Recommended for ML)
```bash
# Set start command to:
gunicorn app:app --bind 0.0.0.0:$PORT
```

> **Note:** TensorFlow is ~450MB. For Vercel serverless, use `tensorflow-cpu` and the `excludeFiles` config in `vercel.json`. For zero-size-limit deployment, host the Flask backend on **Render** (free tier) and the frontend on **Vercel**.

---

## ✦ Tech Stack

| Layer | Technology |
|---|---|
| **Deep Learning** | TensorFlow 2.21, Keras 3.14 |
| **CNN Backbone** | ResNet50 (ImageNet pretrained) |
| **Temporal Model** | 2-Layer LSTM (256 → 128 units) |
| **Video Processing** | OpenCV 4.13 |
| **Backend API** | Flask 3.1, Gunicorn |
| **Frontend** | Vanilla HTML/CSS/JS (no frameworks) |
| **Animations** | CSS keyframes + Canvas API |
| **Data Science** | NumPy, Pandas, Scikit-learn, Seaborn |
| **Dataset** | UCF-101 (20 classes) |

---

## ✦ Dependencies

```
tensorflow>=2.21
opencv-python>=4.8
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
scikit-learn>=1.3
seaborn>=0.13
tqdm>=4.65
flask>=3.0
werkzeug>=3.0
gunicorn>=21.0
```

---

## ✦ Results

- **Architecture:** ResNet50 feature extractor + 2-layer LSTM
- **Training Data:** UCF-101 (20 selected action classes)
- **Parameters:** 2,567,508 total
- **Input:** 16 frames × 2048 CNN features = `(16, 2048)` sequence
- **Output:** Softmax probabilities over 20 classes

---

<div align="center">

<br/>

**Made with ❤️ and deep learning**

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1a1a,100:000000&height=100&section=footer" width="100%"/>

</div>
