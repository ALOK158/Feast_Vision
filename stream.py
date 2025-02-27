# Add warning suppressions at the top
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import streamlit as st
import numpy as np
import gdown
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# Force TensorFlow to use CPU on Streamlit Cloud
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Model configuration
MODEL_ID = "12_1pkbE5zjeySCrgkHGjghiIEgPhQHXn"
MODEL_PATH = "FinalModel_1.keras"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={MODEL_ID}"

# Set page config
st.set_page_config(
    page_title="FEAST AI",
    page_icon="🍕",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .stAlert {
            padding: 1rem;
            margin: 1rem 0;
        }
        .prediction-box {
            padding: 2rem;
            border-radius: 0.5rem;
            background-color: #f0f2f6;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_trained_model():
    try:
        if not os.path.exists(MODEL_PATH):
            with st.spinner("📥 Downloading model... This may take a while."):
                gdown.download(MODEL_URL, MODEL_PATH, fuzzy=True, quiet=False)
        
        with st.spinner("🔄 Loading model..."):
            try:
                # First try loading without custom objects
                model = load_model(MODEL_PATH)
                return model
            except:
                # If that fails, try loading with compile=False
                model = load_model(MODEL_PATH, compile=False)
                # Recompile the model with basic settings
                model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.error("Please check if the model file is corrupted or incompatible.")
        return None

def preprocess_image(image):
    try:
        image = image.convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"❌ Error preprocessing image: {str(e)}")
        return None

def main():
    st.title("🍕 FEAST AI")
    st.write("Upload Image of Food you find difficult to recognize")
    
    model = load_trained_model()
    if model is None:
        st.error("🚨 Model failed to load. Please check logs.")
        return

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Supported formats: JPG, JPEG, PNG, BMP"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            with st.spinner("🔄 Processing image..."):
                start_time = time.time()
                processed_image = preprocess_image(image)
                
                if processed_image is not None:
                    try:
                        prediction = model.predict(processed_image, verbose=0)
                        predicted_class_index = np.argmax(prediction, axis=1)[0]
                        processing_time = time.time() - start_time
                        
                        st.markdown("### 📊 Prediction Results")
                        st.markdown(
                            f"""
                            <div class="prediction-box">
                                <h4>Predicted Class Index: {predicted_class_index}</h4>
                                <p>Processing Time: {processing_time:.3f} seconds</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()
