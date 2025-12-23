import streamlit as st
import cv2
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import tempfile
import os
import numpy as np

# =========================
# Streamlit App Config
# =========================
st.set_page_config(page_title="Triple Riding Detection", layout="wide")
st.title("🚨 Triple Riding Detection System (YOLOv8 + Streamlit)")

# =========================
# Load YOLOv8 Model
# =========================
MODEL_PATH = "models/best.pt"
if not os.path.exists(MODEL_PATH):
    st.error("❌ best.pt not found! Please place your trained model in the 'models/' folder.")
    st.stop()

model = YOLO(MODEL_PATH)
st.sidebar.success("✅ Model Loaded Successfully!")

# =========================
# Violation CSV Setup
# =========================
CSV_PATH = "data/violations.csv"
if not os.path.exists("data"):
    os.makedirs("data")

if not os.path.exists(CSV_PATH):
    df = pd.DataFrame(columns=["timestamp", "frame_id", "rider_count", "violation_type"])
    df.to_csv(CSV_PATH, index=False)

# =========================
# Triple Riding Detection Function
# =========================
def detect_triple_riding(frame, frame_id=0):
    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])
        if conf < 0.4:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if label.lower() == "triple_riding":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_row = pd.DataFrame([[timestamp, frame_id, "-", "Triple Riding"]],
                                   columns=["timestamp", "frame_id", "rider_count", "violation_type"])
            new_row.to_csv(CSV_PATH, mode='a', header=False, index=False)
        
        # --- HELMET VIOLATION ---
        elif label.lower() in ["without_helmet", "without_helmets"]:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_row = pd.DataFrame([[timestamp, frame_id, "-", "Helmet Violation"]],
                                   columns=["timestamp", "frame_id", "rider_count", "violation_type"])
            new_row.to_csv(CSV_PATH, mode='a', header=False, index=False)
       
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

# =========================
# Streamlit Sidebar Options
# =========================
st.sidebar.header("⚙️ Detection Mode")
mode = st.sidebar.radio("Choose Input Source:", ["🖼️ Image", "📁 Upload Video", "🎥 Webcam"])

# =========================
# IMAGE Mode
# =========================
if mode == "🖼️ Image":
    uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        frame = cv2.resize(frame, (640, 480))
        annotated = detect_triple_riding(frame)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

# =========================
# VIDEO Upload Mode
# =========================
elif mode == "📁 Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        st.sidebar.info("✅ Video uploaded successfully!")
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            frame = cv2.resize(frame, (640, 480))
            annotated = detect_triple_riding(frame, frame_id)
            stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                          channels="RGB", use_column_width=True)

        cap.release()
        os.unlink(tfile.name)

# =========================
# WEBCAM Mode
# =========================
elif mode == "🎥 Webcam":
    cap = cv2.VideoCapture(0)
    st.sidebar.warning("🎦 Press 'Stop Webcam' to end the session.")
    stframe = st.empty()
    frame_id = 0
    stop_btn = st.sidebar.button("Stop Webcam")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop_btn:
            break
        frame_id += 1
        frame = cv2.resize(frame, (640, 480))
        annotated = detect_triple_riding(frame, frame_id)
        stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                      channels="RGB", use_column_width=True)

    cap.release()

# =========================
# Display Violation Logs
# =========================
st.subheader("📋 Detected Violations Log")
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    st.dataframe(df)

    if not df.empty:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Violations CSV",
            data=csv,
            file_name="violations.csv",
            mime="text/csv"
        )
else:
    st.info("No violations recorded yet.")
