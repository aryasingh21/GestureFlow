"""
demo.py  —  GestureFlow Cloud Demo
====================================
Upload a hand image → model predicts the gesture.
This is the cloud-deployable version of GestureFlow.
The full app (with live webcam) runs locally via app.py.
"""

import streamlit as st
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="GestureFlow Demo", page_icon="🖐", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', sans-serif;
    background: #0c0c14 !important; color: #e8e4dc;
}
[data-testid="stAppViewContainer"] > .main { background: transparent !important; }
[data-testid="stHeader"], footer, #MainMenu { display: none !important; }
.title {
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 42px;
    background: linear-gradient(120deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 4px;
}
.sub { text-align: center; color: rgba(255,255,255,0.35);
       font-size: 13px; letter-spacing: .12em; margin-bottom: 32px; }
.result-box {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px; padding: 24px; text-align: center; margin-top: 20px;
}
.gesture-label {
    font-family: 'Syne', sans-serif; font-size: 36px; font-weight: 800;
    margin-bottom: 8px;
}
.conf { font-size: 14px; color: rgba(255,255,255,0.4); margin-bottom: 16px; }
.info-card {
    background: rgba(167,139,250,0.08); border: 1px solid rgba(167,139,250,0.2);
    border-radius: 12px; padding: 16px 20px; margin-bottom: 16px;
    font-size: 13px; color: rgba(255,255,255,0.6); line-height: 1.8;
}
.info-card strong { color: #a78bfa; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">GestureFlow</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">DEEP LEARNING · HAND GESTURE · SLIDE CONTROL</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
<strong>What is GestureFlow?</strong><br>
GestureFlow uses a Convolutional Neural Network (CNN) trained on hand images to recognise 
3 gestures in real time via webcam — controlling PowerPoint slides without a clicker.<br><br>
<strong>Gestures:</strong> &nbsp;
☞ <strong>next</strong> (point right → next slide) &nbsp;|&nbsp;
☜ <strong>prev</strong> (point left → prev slide) &nbsp;|&nbsp;
🤌 <strong>exit</strong> (open palm → close)
</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        import tensorflow as tf
        from tensorflow import keras
        if os.path.exists("gesture_model.h5"):
            m = keras.models.load_model("gesture_model.h5")
            return m
        return None
    except Exception:
        return None

model = load_model()
CLASS_NAMES = ['exit', 'next', 'prev']
GESTURE_INFO = {
    'next': ('>> NEXT SLIDE', '#4ade80', '☞ Index finger pointing RIGHT'),
    'prev': ('<< PREV SLIDE', '#fb923c', '☜ Index finger pointing LEFT'),
    'exit': ('OPEN PALM — EXIT',  '#a78bfa', '🤌 Thumb + index finger touching'),
}

# ── Upload section ────────────────────────────────────────────────────────────
st.markdown("### Try it — Upload a hand image")
uploaded = st.file_uploader(
    "Upload a photo of your hand showing one of the 3 gestures",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img, caption="Your uploaded image", use_column_width=True)

    with col2:
        if model is None:
            st.error("Model not available in this demo.")
        else:
            # Preprocess exactly as training: resize to 64x64, normalise
            img_resized = img.resize((64, 64))
            img_array  = np.array(img_resized).astype(np.float32) / 255.0
            img_array  = img_array.reshape(1, 64, 64, 3)

            probs      = model.predict(img_array, verbose=0)[0]
            top_idx    = int(np.argmax(probs))
            confidence = float(probs[top_idx])
            gesture    = CLASS_NAMES[top_idx]

            label, color, desc = GESTURE_INFO.get(gesture, (gesture, '#ffffff', ''))

            st.markdown(f"""
            <div class="result-box">
                <div class="gesture-label" style="color:{color}">{label}</div>
                <div class="conf">Confidence: {int(confidence*100)}%</div>
                <div style="font-size:14px;color:rgba(255,255,255,0.5)">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence bars for all classes
            st.markdown("<br>**All class probabilities:**", unsafe_allow_html=True)
            for i, (cls, prob) in enumerate(zip(CLASS_NAMES, probs)):
                st.progress(float(prob), text=f"{cls}: {int(prob*100)}%")

# ── How it works ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### How It Works")
st.markdown("""
<div class="info-card">
<strong>Pipeline:</strong><br>
Webcam frame → MediaPipe detects hand → Crop hand region → Resize to 64×64 → 
CNN predicts gesture → PyAutoGUI sends arrow key → PowerPoint advances<br><br>
<strong>Model:</strong> CNN with 2 Conv layers + 2 Dense layers trained on self-collected hand images<br>
<strong>Accuracy:</strong> 94–98% on validation set (with data augmentation)<br>
<strong>Dataset:</strong> 500 images per gesture (300 original + 200 augmented)
</div>
""", unsafe_allow_html=True)

st.markdown("### Run Locally (Full Webcam Version)")
st.code("""git clone https://github.com/aryasingh21/GestureFlow
cd GestureFlow
pip install -r requirements.txt
python dataset_capture.py   # collect your own data
python train_model.py        # train the model
streamlit run app.py         # launch with webcam""", language="bash")

st.markdown("""
<div style="text-align:center;margin-top:32px;font-size:12px;color:rgba(255,255,255,0.2)">
GestureFlow — B.Tech Deep Learning Project &nbsp;·&nbsp; 
<a href="https://github.com/aryasingh21/GestureFlow" style="color:#a78bfa">GitHub</a>
</div>
""", unsafe_allow_html=True)
