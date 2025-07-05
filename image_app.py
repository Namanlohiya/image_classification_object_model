from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import webbrowser
from threading import Timer
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MODEL_PATH'] = 'image_classifier\models\efficientnet_model.h5'  # Update with your model path
app.config['CLASS_NAMES'] = ['bird', 'cat', 'dog']  # Update with your classes

# Function to open browser
def open_browser():
    webbrowser.open_new('http://localhost:5000')

# Load model with error handling
def load_model_safely():
    try:
        model = load_model(app.config['MODEL_PATH'])
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("\nPossible solutions:")
        print("1. Run train.py to create a model")
        print("2. Place model.h5 in the models/ folder")
        print("3. Check the model path is correct")
        exit(1)

model = load_model_safely()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image file")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    
    predictions = model.predict(img)[0]
    top_idx = np.argmax(predictions)
    
    return {
        'class': app.config['CLASS_NAMES'][top_idx],
        'confidence': float(predictions[top_idx]),
        'all_predictions': dict(zip(app.config['CLASS_NAMES'], predictions))
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            
            result = predict_image(filename)
            
            return jsonify({
                'success': True,
                'prediction': result['class'],
                'confidence': f"{result['confidence']*100:.2f}%",
                'image_url': filename
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # Open browser after 1 second
    Timer(1, open_browser).start()
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)