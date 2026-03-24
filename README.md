# GestureFlow 🖐

### Deep Learning-Based Hand Gesture Recognition for Presentation Slide Control

Control PowerPoint slides using hand gestures detected via your webcam — no clicker needed.

[gestureflow-5jed74zbad9txbrjdxtkzk.streamlit.app](https://gestureflow-g9yr9wlux6hhbbtapw9vbu.streamlit.app/)

---

## Gestures

| Gesture  | How to do it                                 | Action                         |
| -------- | -------------------------------------------- | ------------------------------ |
| **next** | Point index finger to the RIGHT →            | Next slide                     |
| **prev** | Point index finger to the LEFT ←             | Previous slide                 |
| **exit** | Open palm — hold for 0.5 secs | Close PowerPoint + stop camera |

---

## Tech Stack

| Component           | Technology                                 |
| ------------------- | ------------------------------------------ |
| Deep Learning Model | TensorFlow / Keras (CNN)                   |
| Hand Detection      | MediaPipe (bounding box crop)              |
| UI                  | Streamlit                                  |
| Webcam Capture      | OpenCV                                     |
| Slide Control       | PyAutoGUI + PyGetWindow                    |
| Data Augmentation   | OpenCV (rotation, brightness, zoom, noise) |

---

## Project Structure

```
GestureFlow/
├── app.py                  # Main Streamlit application
├── dataset_capture.py      # Collect training data from webcam
├── train_model.py          # Train the CNN model
├── augment_dataset.py      # Augment dataset for better generalisation
├── requirements.txt        # Python dependencies
├── GestureFlow.ipynb       # Full project notebook (report)
└── dataset/                # Not included — collect your own
    ├── exit/
    ├── next/
    └── prev/
```

---

## Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect dataset

```bash
python dataset_capture.py
```

- Press `1` → exit gesture (open palm)
- Press `2` → next gesture (point right)
- Press `3` → prev gesture (point left)
- Collect 300 photos per gesture

### 3. Augment dataset (optional but recommended)

```bash
python augment_dataset.py
```

Generates 500 images per class using rotation, brightness, zoom, and noise augmentation.

### 4. Train model

```bash
python train_model.py
```

### 5. Run the app

```bash
streamlit run app.py
```

### 6. Use it

1. Open PowerPoint → **Slide Show tab → From Beginning**
2. Click **Start Camera** in the Streamlit browser
3. Show gestures to control slides

---

## Model Architecture

```
Input (64×64×3 RGB image — hand crop)
  ↓
Conv2D(32 filters, 3×3, ReLU)
  ↓
MaxPooling2D(2×2)
  ↓
Conv2D(64 filters, 3×3, ReLU)
  ↓
MaxPooling2D(2×2)
  ↓
Flatten
  ↓
Dense(128, ReLU)
  ↓
Dense(3, Softmax)  →  [P(exit), P(next), P(prev)]
```

---

## Results

| Gesture              | Precision | Recall | F1-Score    |
| -------------------- | --------- | ------ | ----------- |
| exit                 | 1.00      | 1.00   | 1.00        |
| next                 | 1.00      | 1.00   | 1.00        |
| prev                 | 1.00      | 1.00   | 1.00        |
| **Overall Accuracy** |           |        | **~96-98%** |

> Note: 100% accuracy on clean data indicates overfitting. With augmentation, accuracy stabilises at 94–98%, which better reflects real-world generalisation.

---

## How It Works

```
Webcam frame
    ↓
MediaPipe detects hand bounding box
    ↓
Crop hand region + resize to 64×64
    ↓
CNN predicts gesture (exit / next / prev)
    ↓
If confidence ≥ 50%:
    PyGetWindow focuses PowerPoint
    PyAutoGUI sends arrow key
    ↓
Slide advances
```

---

## Deployment Note

This application requires a webcam and local desktop access (PyAutoGUI needs a physical screen to control PowerPoint). It runs locally via `streamlit run app.py`. Cloud deployment is not supported due to hardware requirements.

---

## Limitations & Future Work

- **Single subject dataset** — trained on one person, may not generalise to all users
- **Lighting sensitivity** — performance degrades in low or inconsistent lighting
- **No temporal context** — each frame classified independently (no motion tracking)
- **Future**: LSTM for temporal gesture sequences, multi-user dataset, edge device deployment

---

_GestureFlow — B.Tech Deep Learning Project_
