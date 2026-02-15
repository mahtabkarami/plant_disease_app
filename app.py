import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

IMG_SIZE = (160, 160)

st.set_page_config(page_title="Plant Disease Classifier", layout="centered")

st.title("🌿 Plant Disease Classifier")
st.write("MobileNetV2 + metadata fusion")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_fusion.h5")
    return model

model = load_model()

# Load labels
with open("labels.json", "r") as f:
    label_names = json.load(f)

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

# -----------------------------
# METADATA INPUT
# -----------------------------
st.sidebar.header("Environmental Data")

humidity = st.sidebar.slider("Humidity (%)", 0, 100, 60) / 100.0
temperature = st.sidebar.slider("Temperature (°C)", 0, 50, 25) / 40.0
soil_ph = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.5) / 10.0

metadata = np.array([[humidity, temperature, soil_ph]])

# -----------------------------
# PREDICTION
# -----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    if st.button("Predict"):
        preds = model.predict([img, metadata])
        class_idx = np.argmax(preds)
        confidence = np.max(preds)

        st.subheader(f"Prediction: {label_names[class_idx]}")
        st.write(f"Confidence: {confidence*100:.2f}%")
