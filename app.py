from flask import Flask, render_template, request, jsonify
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load your trained models
brain_models = {
    'AlexNet': load_model('brain_models/brain_tumor_AlexNet_model.h5'),
    'GoogleNet': load_model('brain_models/brain_tumor_GoogleNet_model.h5'),
    'LeNet-5': load_model('brain_models/brain_tumor_LeNet-5_model.h5'),
    'VGGNet': load_model('brain_models/brain_tumor_VGGNet_model.h5')
}

chest_models = {
    'AlexNet': load_model('chest_models/chest_AlexNet_model.h5'),
    'GoogleNet': load_model('chest_models/chest_GoogleNet_model.h5'),
    'LeNet-5': load_model('chest_models/chest_LeNet-5_model.h5'),
    'VGGNet': load_model('chest_models/chest_VGGNet_model.h5')
}

# Function to preprocess image
def preprocess_image(img_path, img_size=(128, 128)):
    img = Image.open(img_path).resize(img_size)
    img = img.convert('L')  # Convert to grayscale
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Route to render the upload form
@app.route('/')
def upload_form():
    return render_template('index.html')

# Route to handle file upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Determine which model to use based on user selection
        xray_type = request.form['xray_type']
        model_type = request.form['model_type']
        
        if xray_type == 'brain':
            model = brain_models.get(model_type)
            class_labels = ['Tumor', 'No Tumor']
        elif xray_type == 'chest':
            model = chest_models.get(model_type)
            class_labels = ['Normal', 'Tuberculosis']
        else:
            return jsonify({'error': 'Invalid X-ray type selection'})
        
        if model is None:
            return jsonify({'error': 'Invalid model selection'})
        
        # Preprocess the uploaded image
        processed_img = preprocess_image(file_path)
        
        # Make prediction using the selected model
        predictions = model.predict(processed_img)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class = class_labels[predicted_class_index]
        
        result = {
            'xray_type': xray_type,
            'model_type': model_type,
            'predicted_class': predicted_class
        }
        
        return jsonify(result)
    
    return jsonify({'error': 'Unknown error'})

if __name__ == '__main__':
    app.run(debug=True)