import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load trained model
MODEL_PATH = "FinalModel_1.keras"  # Update with correct file path to your .keras model
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

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
    predicted_class_index = np.max(prediction, axis=1)[0]  # Get the class index with highest probability
    st.write(f"### Predicted Class Index: {predicted_class_index}")
    
    # Display confidence
    confidence = np.max(prediction) * 100
    st.write(f"Confidence: {confidence:.2f}%")
