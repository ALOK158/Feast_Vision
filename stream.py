import streamlit as st
import numpy as np
import gdown
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Function to download model from Google Drive
@st.cache_resource  # Ensures model is downloaded only once
def load_trained_model():
    MODEL_URL = "https://drive.google.com/uc?id=12_1pkbE5zjeySCrgkHGjghiIEgPhQHXn"  # Your model's file ID
    MODEL_PATH = "FinalModel_1.keras"

    if not os.path.exists(MODEL_PATH):  # Download only if not already downloaded
        with st.spinner("Downloading model... This may take a while."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    # Load the model
    return load_model(MODEL_PATH)

# Load the model once
model = load_trained_model()

# Function to preprocess input image
def preprocess_image(image):
    """Preprocess the image before feeding it to the model."""
    image = image.convert("RGB")  # Ensure 3 channels
    image = image.resize((224, 224))  # Resize to match model input
    
    # Convert image to NumPy array
    image = np.array(image, dtype=np.float32)
    image = image / 255.0  # Normalize if required (ensure this matches training)
    
    # Expand dimensions to match model input shape (batch_size, height, width, channels)
    image = np.expand_dims(image, axis=0)
    
    return image

# Streamlit UI
st.title("Image Classification with EfficientNetB0")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    with st.spinner("Processing image..."):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
    
    # Get predicted class index
    predicted_class_index = np.argmax(prediction, axis=1)[0]  # Get the class index with highest probability
    confidence = np.max(prediction) * 100
    
    st.write(f"### Predicted Class Index: {predicted_class_index}")
    st.write(f"Confidence: {confidence:.2f}%")
