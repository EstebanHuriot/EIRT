import cv2
from deepface import DeepFace
import streamlit as st
import numpy as np

st.title("Emotion Detection Demo")

image = st.camera_input("Take a picture")

if image is not None:

    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    try:
        res = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        result = res[0] if isinstance(res, list) else res
        emotion = result.get('dominant_emotion')
        confidence = result.get('face_confidence')
        st.write("Emotion:", emotion)
        st.write("Confidence:", confidence)
        st.bar_chart(result["emotion"])
    
    except Exception:
        st.write("No face detected")

