import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import streamlit as st
import numpy as np
import gdown
from PIL import Image
import tensorflow as tf
import time

# Force TensorFlow to use CPU on Streamlit Cloud
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Model configuration
MODEL_ID = "12_1pkbE5zjeySCrgkHGjghiIEgPhQHXn"
MODEL_PATH = "FinalModel_1.4.keras"
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
        .stAlert { padding: 1rem; margin: 1rem 0; }
        .prediction-box { padding: 2rem; border-radius: 0.5rem; background-color: #f0f2f6; margin: 1rem 0; }
        .stButton { margin: 0.5rem 0; }
        .recipe-box { padding: 1rem; border-left: 4px solid #4CAF50; background-color: #e8f5e9; margin-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_trained_model():
    try:
        if not os.path.exists(MODEL_PATH):
            with st.spinner("📥 Downloading model... This may take a while."):
                gdown.download(MODEL_URL, MODEL_PATH, fuzzy=True, quiet=False)
        
        with st.spinner("🔄 Loading model..."):
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            # Verify model output shape
            output_shape = model.output_shape[-1]
            if output_shape != 101:
                st.warning(f"⚠️ Model output shape is {output_shape}, expected 101. Predictions may be incorrect.")
            return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.error("Please try again later or check the model file.")
        return None

def preprocess_image(image):
    try:
        image = image.convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image, dtype=np.float16) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        # Log image stats for debugging
        st.write(f"📷 Image shape: {img_array.shape}, Min: {img_array.min():.2f}, Max: {img_array.max():.2f}")
        return img_array
    except Exception as e:
        st.error(f"❌ Error preprocessing image: {str(e)}")
        return None

def get_top_k_predictions(prediction, food_labels, k=3):
    """Return top k predicted classes with their confidence scores."""
    probs = np.squeeze(prediction)
    top_k_indices = np.argsort(probs)[-k:][::-1]
    top_k_labels = [food_labels[i] if i < len(food_labels) else "Unknown" for i in top_k_indices]
    top_k_confidences = [probs[i] * 100 for i in top_k_indices]
    return list(zip(top_k_labels, top_k_confidences))

def main():
    st.title("🍕 FEAST AI")
    st.write("Upload an image of food you find difficult to recognize!")

    model = load_trained_model()
    if model is None:
        st.button("Retry Loading Model", on_click=lambda: st.experimental_rerun())
        return

    # Food labels for Food-101 dataset
    food_labels = [
        "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare", "beet_salad", 
        "beignets", "bibimbap", "bread_pudding", "breakfast_burrito", "bruschetta", "caesar_salad", 
        "cannoli", "caprese_salad", "carrot_cake", "ceviche", "cheesecake", "cheese_plate", 
        "chicken_curry", "chicken_quesadilla", "chicken_wings", "chocolate_cake", "chocolate_mousse", 
        "churros", "clam_chowder", "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", 
        "cup_cakes", "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots", 
        "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries", "french_onion_soup", 
        "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi", 
        "greek_salad", "grilled_cheese_sandwich", "grilled_salmon", "guacamole", "gyoza", "hamburger", 
        "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna", 
        "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup", 
        "mussels", "nachos", "omelette", "onion_rings", "oysters", "pad_thai", "paella", "pancakes", 
        "panna_cotta", "peking_duck", "pho", "pizza", "pork_chop", "poutine", "prime_rib", 
        "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi", 
        "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese", "spaghetti_carbonara", 
        "spring_rolls", "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu", 
        "tuna_tartare", "waffles"
    ]
    
    # Nutrition and recipes (minimal as in original)
    nutrition = {"pizza": "~800 kcal", "sushi": "~200 kcal", "hamburger": "~600 kcal", "steak": "~700 kcal"}
    recipes = {
        "pizza": "Margherita Pizza: Ingredients - Tomato, Mozzarella, Basil; Instructions - Spread sauce, add cheese, bake at 200°C for 15 min.",
        "sushi": "California Roll: Ingredients - Sushi rice, Avocado, Crab; Instructions - Roll with seaweed, slice into 8 pieces."
    }

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
                        # Log prediction stats for debugging
                        st.write(f"🔍 Prediction shape: {prediction.shape}, Max prob: {np.max(prediction):.4f}, Min prob: {np.min(prediction):.4f}")
                        
                        predicted_class_index = np.argmax(prediction, axis=1)[0]
                        confidence = np.max(prediction) * 100
                        
                        # Validate predicted index
                        if predicted_class_index >= len(food_labels):
                            st.error(f"❌ Predicted index {predicted_class_index} is out of bounds for {len(food_labels)} classes.")
                            predicted_food = "Unknown"
                        else:
                            predicted_food = food_labels[predicted_class_index]
                        
                        processing_time = time.time() - start_time
                        
                        # Get top 3 predictions
                        top_k_preds = get_top_k_predictions(prediction, food_labels, k=3)
                        
                        st.markdown("### 📊 Prediction Results")
                        st.markdown(
                            f"""
                            <div class="prediction-box">
                                <h4>Predicted Food: {predicted_food}</h4>
                                <p>Confidence: {confidence:.1f}%</p>
                                <p>Processing Time: {processing_time:.3f} seconds</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Display top 3 predictions
                        st.markdown("#### Top 3 Predictions")
                        for label, conf in top_k_preds:
                            st.write(f"- {label}: {conf:.1f}%")
                        
                        if predicted_food in recipes:
                            st.markdown(
                                f"""
                                <div class="recipe-box">
                                    <h5>🍽️ Recipe Suggestion</h5>
                                    <p>{recipes[predicted_food]}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        if predicted_food in nutrition:
                            st.write(f"🥗 Estimated Nutrition: {nutrition[predicted_food]}")
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                        st.button("Retry Prediction", on_click=lambda: st.experimental_rerun())

if __name__ == "__main__":
    main()
