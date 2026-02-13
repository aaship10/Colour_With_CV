# 🎨 Colour With CV (OpenCV + MediaPipe)

An interactive Air Drawing Application that allows users to draw on the screen using hand gestures captured via a webcam. Built using OpenCV, MediaPipe, and NumPy, this project enables color selection, variable brush thickness, and an eraser mode — all controlled using hand movements.


### 🚀 Features

✋ Hand tracking using MediaPipe Hands

🖊️ Draw using index finger

🎨 Select from multiple colors

📏 Dynamic brush thickness (based on finger distance)

🧽 Eraser mode (activated with left-hand fist)

🖼️ Custom background image

🧼 Clear canvas with keyboard shortcut


### 🛠️ Technologies Used
Python
OpenCV
MediaPipe
NumPy


### 📦 Installation

1️⃣ Clone the Repository
```
git clone https://github.com/aaship10/Colour_With_CV.git
cd Colour_With_CV
```
2️⃣ Install Dependencies
```
pip install opencv-python mediapipe numpy
```
3️⃣ Add Background Image

Place your background image (e.g., butterfly.webp) in the project directory.

▶️ How to Run
```
python paint_and_cv.py
```

Make sure your webcam is connected.


## 🎮 Controls & Gestures

### ✍️ Drawing Mode

Use right-hand index finger to draw.

Brush thickness:

1. Only Right Index Finger → Thin line

2. Right Index Finger and Right Middle Finger → Thick line

### 🎨 Color Selection

Touch the colored rectangles at the bottom of the screen using your right index finger.

Available Colors:

Blue

Green

Red

Cyan

Pink

Yellow

### 🧽 Eraser Mode

Make a fist with your left hand

Move your right index finger to erase

### ⌨️ Keyboard Shortcuts

Press c → Clear canvas

Press q → Quit application

## 🧠 How It Works

MediaPipe Hands detects and tracks up to two hands.

Landmark detection identifies finger positions.

Distance between index and middle fingers controls brush thickness.

Left-hand fist gesture activates eraser mode.

Drawings are rendered onto a transparent canvas and blended with the background.
