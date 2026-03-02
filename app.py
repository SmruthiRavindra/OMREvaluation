import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('models/bubble_model.keras')

st.title("🎯 Handy Evaluation Engine")

uploaded_file = st.file_uploader("Upload Full OMR Sheet", type=['jpg', 'png'])

if uploaded_file:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, channels="BGR", caption="Uploaded OMR Sheet", width=300)

    # Simple Grid Crop (Adjust these numbers to match your sheet rows/cols)
    # This slices the image so the AI looks at individual circles
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Display Results
    st.subheader("Grading Results:")
    
    # We will just show the first bubble as a test
    # In a real run, you'd loop through all Q1-Q10 bubbles here
    test_bubble = cv2.resize(image, (64, 64)) / 255.0
    test_bubble = np.expand_dims(test_bubble, axis=0)
    
    prediction = model.predict(test_bubble)[0][0]
    
    if prediction > 0.5:
        st.success(f"✅ MARKED DETECTED (Confidence: {prediction:.2%})")
    else:
        st.error(f"⭕ NO MARK DETECTED (Confidence: {(1-prediction):.2%})")