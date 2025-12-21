# 🚨 Triple Riding Detection System (YOLOv8 + Streamlit)   

An AI-powered **traffic violation detection system** that identifies **triple riding** and **helmet violations** in real time from **images, videos, or webcam feed** — built using **YOLOv8** and **Streamlit**.

## ⚙️ Algorithm Used: YOLOv8 (You Only Look Once — Version 8)

Your project — the Triple Riding Detection System — uses YOLOv8, a deep learning–based object detection algorithm developed by Ultralytics.
---

## 🧠 1️⃣ What is YOLOv8?

YOLO (You Only Look Once) is a real-time object detection algorithm that can identify and locate multiple objects in an image in a single glance.

-It divides the image into small grid cells.
-Each cell predicts:
    -Bounding boxes (where objects are located)
    -Class labels (what the objects are)
    -Confidence scores (how sure the model is)

So, unlike older algorithms that process an image many times, YOLO looks at it only once — that’s why it’s fast enough for real-time detection like your app. 🚀

## 🤖 Project Objective

To build an AI-based intelligent traffic surveillance system that detects:

-🚴‍♂️ Triple riding on a two-wheeler

-🪖 Helmet violations

-📊 Records every violation with time and date
## 🎯 Project Overview

This project uses **computer vision and deep learning** to automatically detect:
- 🏍️ **Triple Riding** (three people on a two-wheeler)
- ⛑️ **Without Helmet** or **Helmet Violations**
- ✅ Normal riders (for reference)

All detections are logged automatically into a **CSV file** (`violations.csv`) with:
- Timestamp
- Frame ID
- Violation Type

You can later **download** this file for analysis or reporting.

## 🏗️ System Architecture (Explain Simply)

Input (Camera / Image / Video)
        ↓
YOLOv8 AI Model (Detects objects)
        ↓
Violation Detection Logic
        ↓
Visualization (Streamlit Dashboard)
        ↓
Data Logging (CSV for reports)


---

## 🧠 Key Features

| Feature | Description |
|----------|-------------|
| 🖼️ Image Detection | Upload a single image and view detected violations instantly. |
| 🎥 Video Analysis | Upload video footage and analyze frame-by-frame detections. |
| 📸 Live Webcam Detection | Detect violations in real-time from your webcam feed. |
| 🧾 CSV Logging | Automatically logs violations (time, frame, type) into `violations.csv`. |
| 📊 Streamlit Dashboard | Easy-to-use visual interface for non-technical users. |

---

## 🧩 Tech Stack

- **YOLOv8 (Ultralytics)** → For object detection  
- **OpenCV** → For image and video frame processing  
- **Streamlit** → For the interactive web interface  
- **Pandas** → For data logging and CSV handling  
- **Python 3.9+**  

---

ANAGHA READ FROM HERE 

GO TO VS CODE CLICK ON FILE ONTOP LEFT THEN CLICK ON OPEN FOLDER SELECT THIS FOLDER 
FOLDER NAME IS "triple_riding_helmet_riding" 
after opening this folder in vs code 
ON TOP LEFT CORNER YOU WILL SEE THREE DOTS CLICK ON IT SELECT TERMINAL AFTER THAT 
CLICK ON NEW TERMINAL ONCE DONE YOU WILL SEE TERMINAL
ON THE RIGHT SIDE OF THE TERMINAL CLICK ON THE DOWN ARROW MARK NEXT TO  + MARK SELECT CMD 
AFTER THAT COPY PASTE THE BELOW COMMANDS ONE BY ONE MAKE SURE TO HAVE GOOD INTERNET CONNECTION 

pip install ultralytics
pip install streamlit
pip install opencv-python
pip install opencv-python
pip install numpy
pip install pillow
pip install torch torchvision torchaudio
pip install matplotlib
pip install seaborn


one line command to install them 
pip install ultralytics streamlit opencv-python pandas pillow numpy torch torchvision torchaudio matplotlib seaborn
  

after the installation restart the vs code and open CMD terminal once again 

then paste this code 

streamlit run app.py 

this will open it in chrome browser enjoy your project is done 

python -m streamlit run app.py 
