import os
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from inference_nano import AegisRadNano
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Initialize engine globally (Warmstart)
engine = AegisRadNano()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    start_time = time.time()
    try:
        results = engine.run_inference(filepath)
        runtime = time.time() - start_time
        results['runtime'] = f"{runtime:.1f}s"
        results['image_url'] = url_for('static', filename=f'../uploads/{filename}')
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Listen on all interfaces for network access (tablet/phone)
    # Changed to 5001 to avoid macOS 'AirPlay Receiver' port 5000 conflict
    app.run(host='0.0.0.0', port=5001, debug=False)
