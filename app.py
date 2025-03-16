from flask import Flask, render_template, request
from tensorflow.keras.models import load_model # type: ignore
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

app = Flask(__name__)

model = load_model('models/imageclassifier.h5')

UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image-pred' not in request.files:
        return "No file uploaded!", 400  
    
    file = request.files['image-pred']
    
    if file.filename == '':
        return "No selected file!", 400  

    # Save uploaded file to 'static/uploads/'
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Read image using OpenCV
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = cv2.resize(img, (256, 256))  
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)  

    # Model Prediction
    pred = model.predict(img)
    prediction = "Sad" if pred > 0.5 else "Happy"

    return render_template('index.html', image_pred=file.filename, prediction_text=f"Prediction: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)