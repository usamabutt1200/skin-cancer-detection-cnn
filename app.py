import os
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image

# Streamlit template setup for web interface preview
print("--- Skin Cancer Detector Web Deployment Interface ---")

# 1. Model Categories
class_names = ['Benign', 'Malignant']

def load_saved_model():
    """ZIP se extract ki hui .pkl file ko web application ke liye load karna"""
    pkl_path = 'skin_cancer_detector_model.pkl'
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            model = pickle.load(f)
        print("Serialized model loaded successfully!")
        return model
    else:
        print("Error: Model file (.pkl) not found. Please upload it first.")
        return None

def predict_lesion(image_path, model):
    """Nayi image ko process karke predict karna"""
    if model is None:
        return "Model not initialized."
        
    # Image load aur resize karna (Kaggle model matrix shape: 64x64)
    img = Image.open(image_path).resize((64, 64))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Inference / Prediction execution
    predictions = model.predict(img_batch)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class] * 100
    
    return class_names[predicted_class], confidence

# Deployment architecture checklist print karna
print("Application pipeline compiled. Ready for deployment configuration!")
