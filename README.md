<div align="center">

<!-- Animated Header -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:000000,100:1a1a1a&height=200&section=header&text=ActionNet&fontSize=80&fontColor=ffffff&fontAlignY=38&desc=Video%20Action%20Recognition%20%7C%20ResNet50%20%2B%20LSTM%20%26%20VideoMAE&descAlignY=58&descSize=18" width="100%"/>

<br/>

<!-- Badges -->
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.12-ffffff?style=for-the-badge&logo=python&logoColor=black&labelColor=000000)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%20%7C%202.21-ffffff?style=for-the-badge&logo=tensorflow&logoColor=black&labelColor=000000)
![Keras](https://img.shields.io/badge/Keras-2%20%7C%203-ffffff?style=for-the-badge&logo=keras&logoColor=black&labelColor=000000)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ffffff?style=for-the-badge&logo=pytorch&logoColor=black&labelColor=000000)
![Transformers](https://img.shields.io/badge/Transformers-HF-ffffff?style=for-the-badge&logo=huggingface&logoColor=black&labelColor=000000)
![Vercel](https://img.shields.io/badge/Vercel-Frontend-ffffff?style=for-the-badge&logo=vercel&logoColor=black&labelColor=000000)

<br/>

> **A premium, deep learning-powered video understanding system.**
> Built with a dual-engine architecture (Custom ResNet50 + LSTM and State-of-the-Art Vision Transformer VideoMAE).

<br/>

[![Live Production App](https://img.shields.io/badge/🚀%20Live%20Demo-Try%20It%20Now-white?style=for-the-badge&labelColor=000)](https://video-action-recognition-project.vercel.app/)
[![Dataset](https://img.shields.io/badge/📦%20Dataset-UCF--50%20%26%20Kinetics--400-white?style=for-the-badge&labelColor=000)](https://www.kaggle.com/datasets/pevogam/ucf101/data)

</div>

---

## ✦ What is ActionNet?

**ActionNet** is a production-grade, full-stack video action recognition platform. Users can upload video files or capture live streams directly from their webcam, selecting between two distinct, powerful inference engines:

1. **Custom CNN + LSTM Engine:** A ResNet50 spatial feature extractor combined with a custom-trained, multi-layer recurrent network optimized for action sequence tracking.
2. **Vision Transformer (VideoMAE) Engine:** MCG-NJU's state-of-the-art spatio-temporal self-attention network pretrained on Kinetics-400 for high-performance open-domain action recognition.

The application utilizes a decoupled architecture, running a high-speed, modern frontend on **Vercel** connected to a robust, containerized, background-threaded Flask API on **Hugging Face Spaces**.

---

## ✦ Architecture & Execution Flow

```
[ Frontend: Vercel ]
        │  (User uploads video + selects AI Engine)
        ▼
[ POST /predict to Hugging Face Backend ]
        │  (Instantly returns task_id & spawns background thread)
        ├─────────────────────────┐
        ▼                         ▼
[ Background Processing ]    [ Polling /status/<id> ]
  - OpenCV Compression          Every 2 seconds until done
  - Uniform Frame Extraction
  - Inference Execution
        │
        ▼
[ Save results to disk ] ────► [ Complete Status Response ]
                                  │
                                  ▼
                            [ Update UI Dashboard ]
```

---

## ✦ Key Implementation & Troubleshooting Wins

Deploying a complex deep learning pipeline on resource-constrained environments presented unique challenges that we successfully solved:

* **Keras 3 to Keras 2 Deserialization Patch:** Our custom LSTM was trained using Keras 3 on Kaggle, but the target server runs Keras 2 (`tensorflow-cpu==2.15`). We engineered a monkey-patch utility at startup that dynamically translates model layers (`InputLayer`'s `batch_shape` -> `batch_input_shape`) and strips incompatibilities (`quantization_config`, `optional` parameter), allowing seamless compatibility across Keras versions.
* **OpenCV Video Reader Pipeline for Transformers:** Standard Hugging Face transformer pipelines rely on `torchvision.io.read_video` which has deprecation and compatibility conflicts in modern python containers. We bypassed the pipeline class entirely and designed a robust frame reader using **OpenCV** to sample, convert (BGR to RGB), and pad video frames cleanly before sending them directly into the VideoMAE model.
* **Non-Blocking Background Threading:** To prevent HTTP request timeouts during inference, our Flask API responds instantly with a unique `task_id` and processes the ML tasks asynchronously on background worker threads, serving results via status polling.

---

## ✦ Tech Stack

| Layer | Technology |
|---|---|
| **Deep Learning** | PyTorch, TensorFlow 2.15, Keras 2 (patched), Transformers |
| **Model Architectures** | ResNet50 (Transfer Learning) + 2-layer LSTM, VideoMAE (SOTA Transformer) |
| **Video Preprocessing** | OpenCV (CV2) |
| **Backend API** | Flask, Gunicorn, Docker |
| **Frontend UI** | HTML5, CSS3, Vanilla ES6 JavaScript, Canvas API animations |
| **Hosting & Ops** | Vercel (Frontend), Hugging Face Spaces (Backend) |

---

## ✦ Getting Started (Local Run)

### 1. Clone the Project
```bash
git clone https://github.com/7vaibhav31/Video-Action-Recognition-Project.git
cd Video-Action-Recognition-Project
```

### 2. Configure Backend
```bash
cd backend
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

Download your custom `.keras` model and place it under `backend/models/kinetics_best_model.keras`.

Run the server:
```bash
python app.py
```

### 3. Run Frontend
Open `frontend/index.html` in your browser (using Live Server or similar extension), or run it through any static file host. Configure the backend host URL in the JavaScript files to point to `http://localhost:5000`.

---

<div align="center">
Made with ❤️ and Deep Learning.
</div>
