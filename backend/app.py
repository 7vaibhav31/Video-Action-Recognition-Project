import os
import uuid
import json
import threading
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from inference import load_models, predict
from compress import compress_video

# ── App Setup ──────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin API calls from Vercel
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB max upload
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Background Worker ──────────────────────────────────────────────
def async_inference_task(task_id, video_path, model_type='transformer'):
    status_file = os.path.join(app.config['UPLOAD_FOLDER'], f"task_{task_id}.json")
    
    # Generate a filename for the compressed version
    compressed_path = video_path.rsplit('.', 1)[0] + '_compressed.mp4'
    
    try:
        # 1. Compress the video before inference!
        # This limits it to 300 frames and scales it down, fixing sparse sampling and decoding bugs
        print("[app] Starting video compression...")
        is_compressed = compress_video(video_path, compressed_path)
        
        # If compression succeeded, use the compressed video for inference. 
        # Otherwise, fall back to the original video just in case.
        final_video_to_process = compressed_path if is_compressed else video_path
        
        # 2. Run inference on the video
        result = predict(final_video_to_process, model_type=model_type)
        
        if result.get('error'):
            task_data = {'status': 'failed', 'error': result['error']}
        else:
            task_data = {'status': 'complete', 'result': result}
            
    except Exception as e:
        task_data = {'status': 'failed', 'error': str(e)}
        
    finally:
        # Clean up video files to save server space
        for path_to_remove in [video_path, compressed_path]:
            if os.path.exists(path_to_remove):
                try:
                    os.remove(path_to_remove)
                except Exception as e:
                    print(f"[app] Warning: Failed to remove temp video {path_to_remove}: {e}")
                
    # Save result to status file
    try:
        with open(status_file, 'w') as f:
            json.dump(task_data, f)
    except Exception as e:
        print(f"[app] Error writing task result file {status_file}: {e}")


# ── Routes ─────────────────────────────────────────────────────────

@app.route('/')
def index():
    return jsonify({
        'status': 'active',
        'message': 'ActionNet Video Action Recognition API is running.',
        'endpoints': {
            'predict': '/predict (POST)',
            'status': '/status/<task_id> (GET)',
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
    task_id = uuid.uuid4().hex
    unique_filename = f"{task_id}.{ext}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

    try:
        file.save(save_path)
        
        # Write initial processing status file
        status_file = os.path.join(app.config['UPLOAD_FOLDER'], f"task_{task_id}.json")
        with open(status_file, 'w') as f:
            json.dump({'status': 'processing'}, f)
            
        # Get model type
        model_type = request.form.get('model_type', 'transformer')
        
        # Start background thread for prediction
        thread = threading.Thread(target=async_inference_task, args=(task_id, save_path, model_type))
        thread.start()
        
        return jsonify({'task_id': task_id, 'status': 'processing'})
        
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        return jsonify({'error': str(e)}), 500


@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    task_id = secure_filename(task_id)
    status_file = os.path.join(app.config['UPLOAD_FOLDER'], f"task_{task_id}.json")
    
    if not os.path.exists(status_file):
        return jsonify({'error': 'Task not found'}), 404
        
    try:
        with open(status_file, 'r') as f:
            data = json.load(f)
            
        # If task is complete or failed, clean up the status file
        if data.get('status') in ['complete', 'failed']:
            try:
                os.remove(status_file)
            except Exception as e:
                print(f"[app] Warning: Failed to delete status file {status_file}: {e}")
                
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': f'Failed to read task status: {str(e)}'}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'message': 'Video Action Recognition API is running.'})


# ── Startup ────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("[app] Loading models at startup...")
    try:
        load_models()
    except Exception as e:
        print(f"[app] ⚠️ Model loading failed at startup: {e}")
        print("[app] Server will start anyway — models will lazy-load on first request.")
    print("[app] Starting Flask server...")
    app.run(debug=False, host='0.0.0.0', port=5000)
else:
    # When run by gunicorn/Vercel, load models on module import
    # Wrapped in try/except to PREVENT gunicorn worker crash loops
    try:
        load_models()
    except Exception as e:
        print(f"[app] ⚠️ Model loading failed during gunicorn boot: {e}")
        print("[app] Server will start anyway — models will lazy-load on first request.")
