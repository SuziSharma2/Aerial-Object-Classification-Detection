import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load classification model
MODEL_PATH = "../models/mobilenet_best.h5"
clf_model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 224

st.set_page_config(page_title="Aerial Classification", layout="wide")
st.title("🛩️ Bird vs Drone Classification App")

st.write(
    "Upload an image, and the model will classify whether it's a **Bird** or a **Drone**."
)

file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded Image", width=350)

    # Preprocessing
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_arr = np.array(img_resized) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    pred = clf_model.predict(img_arr)[0][0]
    label = "Drone" if pred > 0.5 else "Bird"

    # Show prediction
    st.subheader(f"🔍 Prediction: **{label}**")
    st.progress(float(pred) if pred > 0.5 else 1 - float(pred))

    st.write(f"Confidence Score: **{float(pred):.4f}**")
