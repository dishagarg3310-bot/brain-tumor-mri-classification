import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------------- configuration
IMG_SIZE = 224
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "brain_tumor_final_model.h5"
CLASSES_PATH = MODEL_DIR / "class_names.json"
LOW_CONFIDENCE = 0.60   # below this, the app asks the user to double-check the result

DESCRIPTIONS = {
    "glioma": "A tumor that starts in the glial cells of the brain or spinal cord.",
    "meningioma": "A tumor that grows from the meninges, the membranes covering the brain.",
    "no_tumor": "No tumor pattern was detected in this scan.",
    "pituitary": "A tumor in the pituitary gland, located at the base of the brain.",
}

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="wide")


# ---------------------------------------------------------------- helpers
@st.cache_resource
def load_assets():
    """Load the trained model and the class names once and keep them in memory."""
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(CLASSES_PATH) as f:
        class_names = json.load(f)
    return model, class_names


def preprocess(image: Image.Image) -> np.ndarray:
    """Convert to RGB, resize to 224x224 and scale pixels to 0-1 (same as training)."""
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def pretty(name: str) -> str:
    return name.replace("_", " ").title()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("About")
    st.write(
        "This app classifies a brain MRI scan into one of four categories: "
        "glioma, meningioma, pituitary tumor or no tumor."
    )
    st.subheader("Model")
    st.write("Custom CNN trained from scratch with data augmentation.")
    st.write("Test accuracy: about 82.5%")
    st.write("Weakest class: meningioma (many scans of this type are missed).")
    st.warning(
        "Educational prototype only. It is not a medical device and must not "
        "be used for diagnosis. Always consult a qualified radiologist."
    )

# ---------------------------------------------------------------- main page
st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload a brain MRI image to get the predicted tumor type with confidence scores.")

if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    st.error(
        "Model files not found. Make sure the 'models' folder next to app.py contains "
        "'brain_tumor_final_model.h5' and 'class_names.json'."
    )
    st.stop()

model, class_names = load_assets()

uploaded = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Waiting for an image. Supported formats: JPG, JPEG, PNG.")
else:
    try:
        image = Image.open(uploaded)
        image.load()   # forces Pillow to read the file, so corrupt files fail here
    except Exception:
        st.error("This file could not be read as an image. Please upload a valid JPG or PNG.")
        st.stop()

    with st.spinner("Analyzing the scan..."):
        probs = model.predict(preprocess(image), verbose=0)[0]

    best = int(np.argmax(probs))
    label, confidence = class_names[best], float(probs[best])

    left, right = st.columns(2)
    with left:
        st.subheader("Uploaded scan")
        st.image(image, use_container_width=True)
    with right:
        st.subheader("Prediction")
        st.metric("Predicted class", pretty(label))
        st.metric("Confidence", f"{confidence * 100:.1f}%")
        st.caption(DESCRIPTIONS.get(label, ""))
        if confidence < LOW_CONFIDENCE:
            st.warning("The model is not confident about this scan. Treat the result with caution.")

        st.subheader("Confidence for each class")
        chart_data = pd.DataFrame(
            {"Confidence (%)": probs * 100},
            index=[pretty(c) for c in class_names],
        )
        st.bar_chart(chart_data)