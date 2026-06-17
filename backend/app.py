"""
app.py — Main Flask application for Video Action Recognition.
Serves the dashboard and handles video upload + inference.
"""

import os
import uuid
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from inference import load_models, predict

# ── App Setup ──────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin API calls from Vercel
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB max upload
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Routes ─────────────────────────────────────────────────────────

@app.route('/')
def index():
    return jsonify({
        'status': 'active',
        'message': 'ActionNet Video Action Recognition API is running.',
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)'
        }
    })


@app.route('/predict', methods=['POST'])
def predict_action():
    # Validate file presence
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided.'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'Unsupported format. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    # Save with unique name to avoid collisions
    ext = file.filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

    try:
        file.save(save_path)
        result = predict(save_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up uploaded file after prediction
        if os.path.exists(save_path):
            os.remove(save_path)


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'message': 'Video Action Recognition API is running.'})


# ── Startup ────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("[app] Loading models at startup...")
    load_models()
    print("[app] Starting Flask server...")
    app.run(debug=False, host='0.0.0.0', port=5000)
else:
    # When run by gunicorn/Vercel, load models on module import
    load_models()
