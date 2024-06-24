# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:14:03 2024

@author: Abdelrahman
"""

# api.py

from flask import Flask, request, jsonify
import joblib
from train_model import process_arabic_text

# Load the trained model and other necessary objects
mlp = joblib.load('saved_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Initialize the Flask app
app = Flask(__name__)

# Function to classify a complaint
def classify_complaint(complaint_text):
    processed_text = process_arabic_text(complaint_text)
    complaint_vector = vectorizer.transform([processed_text])
    predicted_label = mlp.predict(complaint_vector)[0]
    predicted_class_name = label_encoder.inverse_transform([predicted_label])[0]
    return predicted_class_name

# Define a route for classification
@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    complaint_text = data['complaint_text']
    predicted_label = classify_complaint(complaint_text)
    return jsonify({'predicted_label': predicted_label})

if __name__ == '__main__':
    app.run()
