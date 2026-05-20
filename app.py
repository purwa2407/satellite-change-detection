import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("forest_fire_model.h5")

st.title("Forest Fire Detection System")

uploaded_file = st.file_uploader(
    "Upload Forest Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    image = image.resize((196, 196))

    img_array = np.array(image) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.error("🔥 Fire Detected")
    else:
        st.success("✅ No Fire Detected")