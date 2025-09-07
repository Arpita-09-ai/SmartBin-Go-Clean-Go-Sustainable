import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime, timedelta
from PIL import Image
import gdown
import os
from translations import translations  # Make sure your translations.py includes 'or' for Odia

# ---------------------------
# Google Drive model
# ---------------------------
DRIVE_ID = "1gjNwy6mzNU4VfEJ9Mb8A7jMwrhQk30Kf"
MODEL_PATH = "predictWaste12.h5"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={DRIVE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    st.success("Model downloaded!")

# ---------------------------
# Load model with caching
# ---------------------------
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

model = load_model(MODEL_PATH)
CLASS_LABELS = [
    "cardboard", "glass", "metal", "paper", "plastic", "trash",
    "clothes", "green-waste", "shoes", "food", "battery", "others"
]

# ---------------------------
# Mediapipe setup
# ---------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---------------------------
# Language selector
# ---------------------------
lang = st.sidebar.selectbox("Select Language / भाषा / ଭାଷା ଚୟନ କରନ୍ତୁ", ["English", "Hindi", "Odia"])
lang_code = {"English": "en", "Hindi": "hi", "Odia": "or"}[lang]
t = translations[lang_code]

# ---------------------------
# Sidebar mode selection
# ---------------------------
st.sidebar.title("♻️ Smart Waste Management")
app_mode = st.sidebar.radio(t["mode_select"], [
    t["upload_predict"], t["live_mediapipe"], t["waste_forecast"]
])

# ---------------------------
# Helper functions
# ---------------------------
def preprocess_image(img):
    img = img.convert("RGB").resize((224,224))
    arr = np.array(img)/255.0
    return np.expand_dims(arr, axis=0)

def preprocess_roi(roi):
    try:
        roi = cv2.resize(roi, (224,224))/255.0
        return np.expand_dims(roi, axis=0)
    except:
        return None

def is_pointing(landmarks):
    tips = [8, 12, 16, 20]
    pip_joints = [6, 10, 14, 18]
    extended = landmarks[tips[0]].y < landmarks[pip_joints[0]].y
    folded = all(landmarks[t].y > landmarks[p].y for t, p in zip(tips[1:], pip_joints[1:]))
    return extended and folded

# ---------------------------
# Mode 1: Upload & Predict
# ---------------------------
if app_mode == t["upload_predict"]:
    st.title(t["upload_predict"])
    uploaded_file = st.file_uploader(t["upload_image"], type=["jpg","jpeg","png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        pred = model.predict(preprocess_image(img))[0]
        idx = np.argmax(pred)
        label = CLASS_LABELS[idx]
        confidence = pred[idx]

        st.success(f"{t['prediction']}: **{label}** ({t['confidence']}: {confidence*100:.2f}%)")

# ---------------------------
# Mode 2: Live Mediapipe Classifier
# ---------------------------
elif app_mode == t["live_mediapipe"]:
    st.title(t["live_mediapipe"])
    run_camera = st.checkbox(t["run_camera"])
    FRAME_WINDOW = st.image([])

    if run_camera:
        cap = cv2.VideoCapture(0)
        pred_buffer = deque(maxlen=5)

        with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.warning(t["no_camera"])
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                h, w, _ = frame.shape

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        if is_pointing(hand_landmarks.landmark):
                            tip = hand_landmarks.landmark[8]
                            wrist = hand_landmarks.landmark[0]
                            cx, cy = int(tip.x * w), int(tip.y * h)
                            wx, wy = int(wrist.x * w), int(wrist.y * h)
                            dx, dy = cx - wx, cy - wy
                            hand_distance = int(np.hypot(dx, dy) * 2)
                            roi_size = max(hand_distance, 50)

                            xmin = max(cx - roi_size//2, 0)
                            ymin = max(cy - roi_size//2, 0)
                            xmax = min(cx + roi_size//2, w)
                            ymax = min(cy + roi_size//2, h)

                            roi = frame[ymin:ymax, xmin:xmax]
                            roi_input = preprocess_roi(roi)
                            if roi_input is not None:
                                pred = model.predict(roi_input, verbose=0)[0]
                                pred_buffer.append(pred)
                                avg_pred = np.mean(pred_buffer, axis=0)
                                top_idx = np.argmax(avg_pred)
                                label = CLASS_LABELS[top_idx]
                                confidence = avg_pred[top_idx]

                                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (255,0,0),2)
                                cv2.putText(frame, f"{label}: {confidence:.2f}", 
                                            (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()

# ---------------------------
# Mode 3: Waste Forecasting (Demo)
# ---------------------------
elif app_mode == t["waste_forecast"]:
    st.title(t["waste_forecast"])

    today = datetime.today()
    dates = [today + timedelta(days=i) for i in range(7)]
    predicted_waste = np.random.randint(15000, 95000, size=7)

    df = pd.DataFrame({"Date": dates, "Predicted_Waste": predicted_waste})
    df["Trucks"] = np.ceil(df["Predicted_Waste"]/50000).astype(int)
    df["Staff"] = df["Trucks"] * 3
    df["Drivers"] = df["Trucks"]

    st.subheader(t["resource_table"])
    st.dataframe(df)

    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["Predicted_Waste"], marker="o")
    ax.set_title("Predicted Daily Waste")
    ax.set_ylabel("Waste (kg)")
    st.pyplot(fig)
