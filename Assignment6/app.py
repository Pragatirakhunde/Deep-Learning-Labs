import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tomato Disease Detector",
    page_icon="🍅",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🍅 Tomato Disease Detector")

st.write(
    "Upload an image of a tomato leaf and the trained CNN "
    "will predict the disease class."
)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = "tomato_disease_cnn.keras"


# ============================================================
# CLASS NAMES
# IMPORTANT:
# These must be in the SAME ORDER as the training dataset.
# ============================================================

class_names = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file '{MODEL_PATH}' not found. "
            "Please place the trained model in the same folder as app.py."
        )
        st.stop()

    model = tf.keras.models.load_model(MODEL_PATH)

    return model


model = load_model()


# ============================================================
# IMAGE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # Open uploaded image
        # ----------------------------------------------------

        image = Image.open(uploaded_file).convert("RGB")


        # ----------------------------------------------------
        # Display image
        # ----------------------------------------------------

        st.subheader("Uploaded Image")

        st.image(
            image,
            caption="Tomato Leaf",
            use_container_width=True
        )


        # ----------------------------------------------------
        # Preprocess image
        # ----------------------------------------------------

        IMG_SIZE = (128, 128)

        resized_image = image.resize(IMG_SIZE)

        image_array = np.array(resized_image)

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        predictions = model.predict(
            image_array,
            verbose=0
        )


        # ----------------------------------------------------
        # Get predicted class
        # ----------------------------------------------------

        predicted_index = np.argmax(
            predictions[0]
        )

        predicted_class = class_names[predicted_index]

        confidence = predictions[0][predicted_index] * 100


        # ----------------------------------------------------
        # Display prediction
        # ----------------------------------------------------

        st.subheader("🔍 Prediction")

        st.success(
            f"Predicted Class: {predicted_class}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )


        # ----------------------------------------------------
        # Display all probabilities
        # ----------------------------------------------------

        st.subheader("📊 Prediction Probabilities")

        probabilities = predictions[0] * 100

        for i in range(len(class_names)):

            st.write(
                f"**{class_names[i]}**: "
                f"{probabilities[i]:.2f}%"
            )

            st.progress(
                float(predictions[0][i])
            )


    except Exception as e:

        st.error(
            f"Error while processing the image: {e}"
        )


# ============================================================
# INFORMATION
# ============================================================

st.divider()

st.subheader("ℹ️ About the Model")

st.write(
    """
    This application uses a Convolutional Neural Network (CNN)
    trained on tomato leaf images.

    The model classifies the input image into one of 10 classes
    including different tomato diseases and healthy leaves.
    """
)

st.caption(
    "Deep Learning Assignment — Tomato Disease Classification"
)