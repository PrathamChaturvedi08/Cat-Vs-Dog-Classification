import streamlit as st
import numpy as np
from PIL import Image
import tf_keras
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense
# from tensorflow.python.keras.models import Sequential, load_model
# from tensorflow.python.keras.layers import LSTM, Dense
import keras
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow_hub as hub
# from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope

with custom_object_scope({'KerasLayer': hub.KerasLayer}):
    model = load_model('cat_dog_classifier.h5')

# Load the trained model
# model = load_model('cat_dog_classifier.h5')

# Function to preprocess the uploaded image
def prepare_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Page title and description
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🐾 Cat vs Dog Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image and let the model tell you whether it's a cat or a dog!</p>", unsafe_allow_html=True)
st.markdown("---")

# File uploader
uploaded_image = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    img = Image.open(uploaded_image)

    # Layout: Image on the left, result on the right
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    with col2:
        img_array = prepare_image(img)
        prediction = model.predict(img_array)

        # Determine class and confidence
        if prediction[0] > 0.5:
            label = "🐶 Dog"
            confidence = prediction[0][0] * 100
        else:
            label = "🐱 Cat"
            confidence = (1 - prediction[0][0]) * 100

        # Highlight result
        st.markdown(f"<h2 style='color: #2E8B57;'>Prediction: {label}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color: #555;'>Confidence: {confidence:.2f}%</h4>", unsafe_allow_html=True)