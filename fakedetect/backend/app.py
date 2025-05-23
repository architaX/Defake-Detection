from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from utils.preprocessing import preprocess_image, preprocess_audio, preprocess_video
from utils.model_loader import load_models
from config import Config
import logging
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models at startup
try:
    models = load_models()
    logger.info("All models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise e

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename, file_type):
    """Check if the file extension is allowed for the given file type"""
    allowed_extensions = {
        'image': {'jpg', 'jpeg', 'png'},
        'audio': {'wav', 'mp3', 'flac'},
        'video': {'mp4', 'mov', 'avi'}
    }
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions.get(file_type, set())

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    file_type = request.form.get('type', 'image')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file or not allowed_file(file.filename, file_type):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Secure filename and save temporarily
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"File saved temporarily at: {filepath}")
        
        # Process based on file type
        if file_type == 'image':
            processed = preprocess_image(filepath)
            prediction = models['image'].predict(processed)
            result = 'Fake' if prediction[0][0] > 0.5 else 'Real'
            confidence = float(prediction[0][0]) if result == 'Fake' else 1 - float(prediction[0][0])
        elif file_type == 'audio':
            processed = preprocess_audio(filepath)
            prediction = models['audio'].predict(processed)
            result = 'Fake' if prediction[0][0] > 0.5 else 'Real'
            confidence = float(prediction[0][0]) if result == 'Fake' else 1 - float(prediction[0][0])
        elif file_type == 'video':
            processed = preprocess_video(filepath)
            prediction = models['video'].predict(processed)
            result = 'Fake' if prediction[0][0] > 0.5 else 'Real'
            confidence = float(prediction[0][0]) if result == 'Fake' else 1 - float(prediction[0][0])
        else:
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'result': result,
            'confidence': round(confidence * 100, 2),
            'file_type': file_type,
            'filename': filename
        })
    
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size limit errors"""
    return jsonify({'error': 'File too large (max 100MB)'}), 413

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)