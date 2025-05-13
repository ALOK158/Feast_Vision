import os
import logging
import streamlit as st
import numpy as np
import gdown
from PIL import Image
import tensorflow as tf
import time



# Force TensorFlow to use CPU on Streamlit Cloud
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Streamlit UI
st.set_page_config(page_title="Image Classifier", layout="centered")  # Set this first

# Model configuration
MODEL_ID = "12_1pkbE5zjeySCrgkHGjghiIEgPhQHXn"
MODEL_PATH = "best_model (2).keras"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={MODEL_ID}"

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

# Load model using the function
model = load_trained_model()

# Define your class names (replace with your actual ones)
class_names = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 
                'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 
                'carrot_cake', 'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 
                'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 
                'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 
                'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 
                'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 
                'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 
                'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 
                'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 
                'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 
                'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 
                'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 
                'tuna_tartare', 'waffles']  # Update this with your actual class names

# Image preprocessing function
def load_and_prep(image_file, shape=224, scale=False):
    image_file.seek(0)  # Reset file pointer
    image = tf.image.decode_image(image_file.read(), channels=3)
    image = tf.image.resize(image, [shape, shape])
    image.set_shape([shape, shape, 3])
    if scale:
        image = image / 255.0
    return image

# Streamlit UI
st.title("🔍 Image Classification App")
st.write("Upload an image to see the top 5 class predictions from the trained model.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Prediction section
    st.write("Classifying...")

    # Preprocess
    img_tensor = load_and_prep(uploaded_file, scale=False)
    pred_prob = model.predict(tf.expand_dims(img_tensor, axis=0))

    # Top-5 Predictions
    top_5_i = (pred_prob.argsort())[0][-5:][::-1]
    top_5_values = pred_prob[0][top_5_i]
    top_5_labels = [class_names[i] for i in top_5_i]

    # Show predictions
    st.subheader("Top 5 Predictions:")
    for label, prob in zip(top_5_labels, top_5_values):
        st.write(f"🔹 **{label}** — Confidence: `{prob:.4f}`")
