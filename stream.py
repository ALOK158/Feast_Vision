import os
import streamlit as st
import gdown
from PIL import Image
import tensorflow as tf
import time
import numpy as np

# Configure the app
st.set_page_config(
    page_title="🍔 FoodVision AI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
    .st-bb {
        background-color: #F0F2F6;
    }
    .st-at {
        background-color: #FFFFFF;
    }
    .st-ax {
        color: #FF4B4B;
    }
    .prediction-card {
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    .prediction-card:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Force TensorFlow to use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Model configuration
MODEL_ID = "1d7tKkgWC0lLooULNYTg_qCLGHbtf98ie"
MODEL_PATH = "food_classifier.keras"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={MODEL_ID}"

# Class names (same as before)
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
               'tuna_tartare', 'waffles']

# Improved model loading with cache
@st.cache_resource
def load_trained_model():
    try:
        if not os.path.exists(MODEL_PATH):
            with st.spinner("🚀 Downloading our food recognition AI... This may take a minute or two"):
                progress_bar = st.progress(0)
                
                def update_progress(current, total, width=80):
                    progress = int(current/total*100)
                    progress_bar.progress(progress)
                
                gdown.download(MODEL_URL, MODEL_PATH, fuzzy=True, quiet=False)
                progress_bar.progress(100)
        
        with st.spinner("🧠 Waking up the food expert..."):
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            model.compile(optimizer='adam', 
                        loss='sparse_categorical_crossentropy', 
                        metrics=['accuracy'])
            return model
    except Exception as e:
        st.error(f"❌ Oops! Our chef is taking a break: {str(e)}")
        st.error("Please try again later or check if the model file is available.")
        return None

# Load model
model = load_trained_model()

# Improved image preprocessing
def load_and_prep(image_file, shape=224):
    image_file.seek(0)
    image = tf.image.decode_image(image_file.read(), channels=3)
    image = tf.image.resize(image, [shape, shape])
    image = image / 255.0  # Normalize to [0,1] range
    return image

# Main app
def main():
    st.title("🍔 FoodVision AI")
    st.markdown("### Discover what's on your plate with AI-powered food recognition!")
    
    with st.expander("ℹ️ How it works"):
        st.write("""
        1. Upload an image of food (photo or screenshot)
        2. Our AI analyzes the visual patterns
        3. Get instant predictions with confidence scores
        """)
        st.image("https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=500", 
                caption="Let's identify your food!", width=300)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📤 Upload Your Food Image")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png"],
            help="For best results, use clear photos of single food items"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Your delicious upload", use_column_width=True)
            
            # Add some fun options
            confidence_threshold = st.slider(
                "Confidence Threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5,
                help="Adjust how confident the AI should be before making predictions"
            )
    
    with col2:
        if uploaded_file and model:
            st.subheader("🔍 Analysis Results")
            
            with st.spinner("👨‍🍳 Chef AI is examining your food..."):
                # Simulate processing time for better UX
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(percent_complete + 1)
                
                # Actual prediction
                img_tensor = load_and_prep(uploaded_file)
                pred_prob = model.predict(tf.expand_dims(img_tensor, axis=0))[0]
                
                # Get top predictions
                top_5_i = pred_prob.argsort()[-5:][::-1]
                top_5_values = pred_prob[top_5_i]
                top_5_labels = [class_names[i] for i in top_5_i]
                
                # Display results in cards
                st.success("✅ Analysis complete!")
                st.balloons()  # Celebration!
                
                for label, prob in zip(top_5_labels, top_5_values):
                    if prob >= confidence_threshold:
                        with st.container():
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h3>🍽️ {label.replace('_', ' ').title()}</h3>
                                <p>Confidence: <strong>{prob*100:.1f}%</strong></p>
                                <progress value="{prob}" max="1"></progress>
                            </div>
                            "", unsafe_allow_html=True)
                
                # Add some fun facts
                winning_food = top_5_labels[0].replace('_', ' ')
                st.markdown(f"**Did you know?** {get_fun_fact(winning_food)}")
                
                # Share button
                st.markdown("### Share your results!")
                st.code(f"My food was identified as {winning_food} with {top_5_values[0]*100:.1f}% confidence!")

# Fun facts dictionary
def get_fun_fact(food_name):
    facts = {
        "apple pie": "The first apple pie recipe was published in 1381!",
        "pizza": "About 3 billion pizzas are sold in the U.S. each year!",
        "sushi": "Real wasabi is rare and expensive - most sushi uses horseradish!",
        "hamburger": "The world's most expensive burger costs $5,000!",
        "tacos": "Tacos have been around since the 18th century in Mexico!"
    }
    return facts.get(food_name.lower(), f"{food_name.title()} is delicious! Want to learn more about it?")

if __name__ == "__main__":
    if model is None:
        st.warning("Model failed to load. Please check the model file.")
    else:
        main()
