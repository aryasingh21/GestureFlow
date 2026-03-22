"""
app.py  —  GestureFlow (Simplified — no freeze/unlock cycle)
=============================================================
CHANGE FROM PREVIOUS VERSION:
  - Removed freeze_mode / ready_for_next / palm-to-unlock logic
  - Replaced with simple per-gesture cooldown timer
  - After a slide action, just wait SWIPE_COOLDOWN seconds, then gesture again
  - Much more practical for real presentations
  - palm gesture now just shows "READY" label — no blocking behaviour
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import os
import time
import base64
import collections
from io import BytesIO
from PIL import Image

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import pygetwindow as gw
    PYGETWINDOW = True
except ImportError:
    PYGETWINDOW = False

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="GestureFlow",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "gesture_model.h5")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

IMG_SIZE             = 64
CONFIDENCE_THRESHOLD = 0.50
SMOOTHING_WINDOW     = 3

# ── Cooldowns: after firing an action, ignore same gesture for this long ──────
SWIPE_COOLDOWN = 1.0   # seconds between slide changes
ZOOM_COOLDOWN  = 0.8
EXIT_HOLD_SEC  = 1.5   # hold exit gesture this long to stop camera

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0.05

PPTX_KEYWORDS = ["powerpoint","impress","presentation","slideshow",
                 "slide show","foxit","adobe","pdf","sumatra"]

def focus_and_press(key):
    if PYGETWINDOW:
        try:
            for title in gw.getAllTitles():
                if any(k in title.lower() for k in PPTX_KEYWORDS):
                    gw.getWindowsWithTitle(title)[0].activate()
                    time.sleep(0.08)
                    break
        except Exception:
            pass
    pyautogui.press(key)

def focus_and_hotkey(*keys):
    if PYGETWINDOW:
        try:
            for title in gw.getAllTitles():
                if any(k in title.lower() for k in PPTX_KEYWORDS):
                    gw.getWindowsWithTitle(title)[0].activate()
                    time.sleep(0.08)
                    break
        except Exception:
            pass
    pyautogui.hotkey(*keys)

# =============================================================================
# CACHED RESOURCES
# =============================================================================
@st.cache_resource
def load_model():
    if not TF_AVAILABLE or not os.path.exists(MODEL_PATH):
        return None
    try:
        m = keras.models.load_model(MODEL_PATH)
        m.predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32), verbose=0)
        return m
    except Exception:
        return None

@st.cache_resource
def get_class_names():
    if not os.path.isdir(DATASET_DIR):
        return sorted(["exit","freeze","next","palm","prev","zoom_in","zoom_out"])
    return sorted([d for d in os.listdir(DATASET_DIR)
                   if os.path.isdir(os.path.join(DATASET_DIR, d))])

@st.cache_resource
def get_mediapipe():
    mp_hands = mp.solutions.hands
    hands    = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.65, min_tracking_confidence=0.55,
    )
    return hands, mp_hands, mp.solutions.drawing_utils

# =============================================================================
# HELPERS
# =============================================================================
def get_hand_crop(frame, hand_landmarks, padding=0.25):
    h, w  = frame.shape[:2]
    lms   = hand_landmarks.landmark
    xs    = [lm.x for lm in lms]
    ys    = [lm.y for lm in lms]
    x_min = max(0.0, min(xs) - padding)
    y_min = max(0.0, min(ys) - padding)
    x_max = min(1.0, max(xs) + padding)
    y_max = min(1.0, max(ys) + padding)
    px1, py1 = int(x_min * w), int(y_min * h)
    px2, py2 = int(x_max * w), int(y_max * h)
    if px2 <= px1 or py2 <= py1:
        return None
    crop = frame[py1:py2, px1:px2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return (crop.astype(np.float32) / 255.0).reshape(1, IMG_SIZE, IMG_SIZE, 3)

def frame_to_b64(frame_bgr):
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode()

def draw_overlay(frame, label, colour, raw_label, confidence, smoothed,
                 last_action, cooldown_remaining):
    h, w = frame.shape[:2]

    # top bar
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w,85), (0,0,0), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)

    if label:
        cv2.putText(frame, label, (14,62),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, colour, 3, cv2.LINE_AA)

    # cooldown bar — shows time remaining before next gesture fires
    if cooldown_remaining > 0:
        bar_w = int(w * cooldown_remaining)
        cv2.rectangle(frame, (0, 82), (bar_w, 88), (100,100,255), -1)
        cv2.putText(frame, f"cooldown  {cooldown_remaining:.1f}s",
                    (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120,120,220), 1)

    # debug panel
    px = w - 265
    cv2.rectangle(frame, (px-4,2), (w-2,112), (10,10,10), -1)
    cv2.rectangle(frame, (px-4,2), (w-2,112), (55,55,55), 1)
    cv2.putText(frame, f"RAW    : {raw_label}",  (px,22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200,200,60), 1)
    cv2.putText(frame, f"SMOOTH : {smoothed}",   (px,40), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80,220,80),  1)
    bar_col = (0,200,80) if confidence >= CONFIDENCE_THRESHOLD else (0,80,220)
    cv2.rectangle(frame, (px,50), (px+220,62), (40,40,40), -1)
    cv2.rectangle(frame, (px,50), (px+int(220*min(confidence,1.0)),62), bar_col, -1)
    cv2.putText(frame, f"CONF   : {int(confidence*100)}%  (min={int(CONFIDENCE_THRESHOLD*100)}%)",
                (px,82), cv2.FONT_HERSHEY_SIMPLEX, 0.38, bar_col, 1)

    # bottom hint
    cv2.putText(frame, "next=→Slide  prev=←Slide  zoom_in/out=Zoom  exit(hold)=Stop",
                (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120,120,120), 1)
    return frame

# =============================================================================
# SESSION STATE
# =============================================================================
for k, v in {
    "running": False,
    "action_log": [],
    "current_label": "—",
    "current_conf": 0.0,
    "next_count": 0,
    "prev_count": 0,
    "last_action_label": "—",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =============================================================================
# CSS
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;800&family=DM+Sans:wght@300;400;500&display=swap');
*, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }
html,body,[data-testid="stAppViewContainer"]{
    font-family:'DM Sans',sans-serif;
    background:#0c0c14 !important; color:#e8e4dc;
}
[data-testid="stAppViewContainer"]>.main{background:transparent !important;}
[data-testid="stHeader"],footer,#MainMenu,
[data-testid="stToolbar"],section[data-testid="stSidebar"]{display:none !important;}
.block-container{padding:1.5rem 2rem !important; max-width:100% !important;}
[data-testid="stVerticalBlock"]{gap:0 !important;}

.gf-title{
    font-family:'Syne',sans-serif;font-weight:800;font-size:26px;
    background:linear-gradient(120deg,#a78bfa,#60a5fa);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    margin-bottom:2px;
}
.gf-sub{font-size:11px;color:rgba(255,255,255,.28);
        letter-spacing:.14em;margin-bottom:18px;}

.cam-frame{border-radius:14px;overflow:hidden;
           border:1px solid rgba(255,255,255,.07);
           background:#050508;width:100%;}
.cam-frame img{width:100%;display:block;border-radius:14px;}
.cam-empty{
    height:420px;display:flex;align-items:center;
    justify-content:center;flex-direction:column;gap:10px;
    border-radius:14px;border:2px dashed rgba(255,255,255,.07);
    background:#050508;color:rgba(255,255,255,.15);
    font-size:13px;letter-spacing:.1em;
}

.stat-row{display:flex;gap:10px;margin-bottom:10px;}
.stat-box{
    flex:1;background:rgba(255,255,255,.04);
    border:1px solid rgba(255,255,255,.07);
    border-radius:10px;padding:10px 14px;
}
.stat-lbl{font-size:9px;font-weight:500;letter-spacing:.18em;
          text-transform:uppercase;color:rgba(255,255,255,.3);margin-bottom:4px;}
.stat-val{font-family:'Syne',sans-serif;font-size:17px;font-weight:600;}
.green{color:#4ade80;} .amber{color:#fbbf24;}
.blue{color:#60a5fa;}  .violet{color:#a78bfa;} .red{color:#f87171;}

.info-box{
    background:rgba(167,139,250,.08);border:1px solid rgba(167,139,250,.2);
    border-radius:10px;padding:12px 16px;font-size:12px;
    color:rgba(255,255,255,.55);line-height:1.75;margin-bottom:12px;
}
.info-box strong{color:#a78bfa;}

.gmap{display:flex;flex-direction:column;gap:5px;margin-bottom:12px;}
.gmap-row{
    display:flex;align-items:center;gap:8px;padding:7px 10px;
    background:rgba(255,255,255,.03);border-radius:7px;
    border-left:2px solid rgba(167,139,250,.35);font-size:12px;
}
.gmap-g{font-family:'Syne',sans-serif;font-weight:600;
        font-size:11px;color:#a78bfa;min-width:85px;}
.gmap-a{color:rgba(255,255,255,.5);}

.log-box{max-height:130px;overflow-y:auto;
         display:flex;flex-direction:column;gap:3px;}
.log-item{font-size:11px;padding:4px 10px;border-radius:6px;
          background:rgba(255,255,255,.02);color:rgba(255,255,255,.4);
          border-left:2px solid rgba(255,255,255,.08);}
.log-item.n{border-left-color:#4ade80;color:#4ade80;}
.log-item.p{border-left-color:#fb923c;color:#fb923c;}
.log-item.z{border-left-color:#60a5fa;color:#60a5fa;}

div[data-testid="stButton"]>button{
    width:100% !important;
    background:linear-gradient(135deg,#7c3aed,#4f46e5) !important;
    color:#fff !important;border:none !important;border-radius:10px !important;
    height:46px !important;font-family:'Syne',sans-serif !important;
    font-size:13px !important;font-weight:700 !important;
    letter-spacing:.1em !important;
    box-shadow:0 4px 20px rgba(124,58,237,.45) !important;
}
.stopbtn div[data-testid="stButton"]>button{
    background:rgba(255,255,255,.06) !important;
    color:rgba(255,255,255,.5) !important;box-shadow:none !important;
}
.warn-box{
    background:rgba(251,191,36,.08);border:1px solid rgba(251,191,36,.25);
    border-radius:8px;padding:10px 14px;font-size:11.5px;
    color:rgba(251,191,36,.8);margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================
st.markdown('<div class="gf-title">GestureFlow</div>', unsafe_allow_html=True)
st.markdown('<div class="gf-sub">DEEP LEARNING · HAND GESTURE · SLIDE CONTROL</div>',
            unsafe_allow_html=True)

cam_col, ctrl_col = st.columns([3, 1], gap="large")

# =============================================================================
# RIGHT PANEL
# =============================================================================
with ctrl_col:

    if not PYGETWINDOW:
        st.markdown("""
        <div class="warn-box">
        ⚠️ Run once:<br>
        <code>pip install pygetwindow</code><br>
        Needed to auto-focus PowerPoint.
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      <strong>How to use:</strong><br>
      1. Open PowerPoint → press <strong>F5</strong><br>
      2. Click <strong>Start Camera</strong><br>
      3. Show gesture → slide changes<br>
      4. Wait for cooldown bar → show next gesture
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.running:
        if st.button("▶  Start Camera"):
            st.session_state.running      = True
            st.session_state.action_log   = []
            st.session_state.next_count   = 0
            st.session_state.prev_count   = 0
            st.session_state.last_action_label = "—"
            st.rerun()
    else:
        st.markdown('<div class="stopbtn">', unsafe_allow_html=True)
        if st.button("⏹  Stop Camera"):
            st.session_state.running = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Live stats
    g    = st.session_state.current_label
    conf = int(st.session_state.current_conf * 100)
    gcol = ("green"  if g in ("next","prev") else
            "blue"   if "zoom" in str(g) else
            "violet" if g in ("palm","open_palm") else "amber")

    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-box">
        <div class="stat-lbl">Gesture</div>
        <div class="stat-val {gcol}">{g}</div>
      </div>
      <div class="stat-box">
        <div class="stat-lbl">Confidence</div>
        <div class="stat-val">{conf}%</div>
      </div>
    </div>
    <div class="stat-row">
      <div class="stat-box">
        <div class="stat-lbl">Last Action</div>
        <div class="stat-val violet">{st.session_state.last_action_label}</div>
      </div>
      <div class="stat-box">
        <div class="stat-lbl">Next / Prev</div>
        <div class="stat-val">{st.session_state.next_count} / {st.session_state.prev_count}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Gesture map
    st.markdown("""
    <div class="gmap">
      <div class="gmap-row"><span class="gmap-g">next</span><span class="gmap-a">Next slide →</span></div>
      <div class="gmap-row"><span class="gmap-g">prev</span><span class="gmap-a">Prev slide ←</span></div>
      <div class="gmap-row"><span class="gmap-g">zoom_in</span><span class="gmap-a">Ctrl +</span></div>
      <div class="gmap-row"><span class="gmap-g">zoom_out</span><span class="gmap-a">Ctrl −</span></div>
      <div class="gmap-row"><span class="gmap-g">exit (hold)</span><span class="gmap-a">Stop camera</span></div>
      <div class="gmap-row" style="opacity:.4"><span class="gmap-g">palm/freeze</span><span class="gmap-a">No action</span></div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.action_log:
        st.markdown("**Recent actions**")
        html = '<div class="log-box">'
        for e in reversed(st.session_state.action_log[-10:]):
            c = "n" if "NEXT" in e else "p" if "PREV" in e else "z"
            html += f'<div class="log-item {c}">{e}</div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

# =============================================================================
# CAMERA FEED
# =============================================================================
with cam_col:
    feed_slot = st.empty()

    if not st.session_state.running:
        feed_slot.markdown("""
        <div class="cam-empty">
          <div style="font-size:44px;opacity:.18">📷</div>
          <div>Press Start Camera to begin</div>
          <div style="font-size:11px;opacity:.5">Make sure PowerPoint slideshow is open (F5)</div>
        </div>""", unsafe_allow_html=True)

    else:
        model       = load_model()
        class_names = get_class_names()
        hands, mp_hands, mp_draw = get_mediapipe()

        if model is None:
            st.error("❌ gesture_model.h5 not found.")
            st.session_state.running = False
            st.stop()

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # ── Per-gesture independent cooldown timers ───────────────────────────
        last_swipe_time = 0.0
        last_zoom_time  = 0.0
        exit_start      = None
        pred_buffer     = collections.deque(maxlen=SMOOTHING_WINDOW)
        next_count      = st.session_state.next_count
        prev_count      = st.session_state.prev_count

        while st.session_state.running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = hands.process(rgb)
            now   = time.time()

            label = None; colour = (255,255,255)
            raw_label = smoothed_label = "no_hand"; confidence = 0.0

            # How long until next swipe is allowed (for progress bar)
            swipe_remaining = max(0.0, SWIPE_COOLDOWN - (now - last_swipe_time))
            cooldown_frac   = swipe_remaining / SWIPE_COOLDOWN

            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style(),
                )

                crop = get_hand_crop(frame, hand)
                if crop is not None:
                    probs      = model.predict(crop, verbose=0)[0]
                    top_idx    = int(np.argmax(probs))
                    confidence = float(probs[top_idx])
                    raw_label  = (class_names[top_idx]
                                  if confidence >= CONFIDENCE_THRESHOLD
                                  and top_idx < len(class_names)
                                  else "low_conf")
                else:
                    raw_label = "low_conf"

                pred_buffer.append(raw_label)
                smoothed_label = max(set(pred_buffer), key=list(pred_buffer).count)

                # ── EXIT: hold to stop ────────────────────────────────────────
                if raw_label == "exit":
                    if exit_start is None:
                        exit_start = now
                    held = now - exit_start
                    prog = min(held / EXIT_HOLD_SEC, 1.0)
                    hf, wf = frame.shape[:2]
                    cv2.rectangle(frame,(0,hf-10),(int(wf*prog),hf),(0,0,220),-1)
                    cv2.putText(frame, f"Hold to Stop  {int(prog*100)}%",
                                (10,hf-16),cv2.FONT_HERSHEY_SIMPLEX,0.6,(60,60,255),2)
                    if held >= EXIT_HOLD_SEC:
    # Close PowerPoint
                        if PYGETWINDOW:
                            try:
                                for title in gw.getAllTitles():
                                    if any(k in title.lower() for k in PPTX_KEYWORDS):
                                        gw.getWindowsWithTitle(title)[0].close()
                                        break
                            except Exception:
                                pass
                        st.session_state.running = False
                        break
                else:
                    exit_start = None

                # ── NEXT SLIDE ────────────────────────────────────────────────
                if smoothed_label == "next":
                    if swipe_remaining == 0:
                        focus_and_press("right")
                        label, colour   = ">> NEXT SLIDE", (80,220,80)
                        last_swipe_time = now
                        next_count     += 1
                        st.session_state.action_log.append(
                            f"{time.strftime('%H:%M:%S')}  NEXT SLIDE")
                        st.session_state.last_action_label = "NEXT →"
                    else:
                        label, colour = ">> NEXT  (wait)", (80,160,80)

                # ── PREV SLIDE ────────────────────────────────────────────────
                elif smoothed_label == "prev":
                    if swipe_remaining == 0:
                        focus_and_press("left")
                        label, colour   = "<< PREV SLIDE", (220,120,0)
                        last_swipe_time = now
                        prev_count     += 1
                        st.session_state.action_log.append(
                            f"{time.strftime('%H:%M:%S')}  PREV SLIDE")
                        st.session_state.last_action_label = "← PREV"
                    else:
                        label, colour = "<< PREV  (wait)", (160,80,0)

                # ── ZOOM IN ───────────────────────────────────────────────────
                elif smoothed_label == "zoom_in":
                    zoom_remaining = max(0.0, ZOOM_COOLDOWN - (now - last_zoom_time))
                    if zoom_remaining == 0:
                        focus_and_hotkey("ctrl","+")
                        label, colour  = "ZOOM IN  +", (80,220,120)
                        last_zoom_time = now
                        st.session_state.action_log.append(
                            f"{time.strftime('%H:%M:%S')}  ZOOM IN")
                        st.session_state.last_action_label = "ZOOM IN"
                    else:
                        label, colour = "ZOOM IN  (wait)", (50,140,80)

                # ── ZOOM OUT ──────────────────────────────────────────────────
                elif smoothed_label == "zoom_out":
                    zoom_remaining = max(0.0, ZOOM_COOLDOWN - (now - last_zoom_time))
                    if zoom_remaining == 0:
                        focus_and_hotkey("ctrl","-")
                        label, colour  = "ZOOM OUT  -", (80,120,220)
                        last_zoom_time = now
                        st.session_state.action_log.append(
                            f"{time.strftime('%H:%M:%S')}  ZOOM OUT")
                        st.session_state.last_action_label = "ZOOM OUT"
                    else:
                        label, colour = "ZOOM OUT  (wait)", (50,80,140)

                # ── PALM / FREEZE — just show label, no blocking ──────────────
                elif smoothed_label in ("palm","open_palm"):
                    label, colour = "PALM", (160,200,160)

                elif smoothed_label == "freeze":
                    label, colour = "FREEZE", (160,160,220)

                elif smoothed_label == "low_conf":
                    label  = f"Low confidence  {int(confidence*100)}%"
                    colour = (100,100,100)

                else:
                    label  = f"{smoothed_label}  {int(confidence*100)}%"
                    colour = (160,160,160)

            else:
                pred_buffer.clear()
                exit_start = None

            frame = draw_overlay(frame, label, colour,
                                 raw_label, confidence, smoothed_label,
                                 None, cooldown_frac)

            feed_slot.markdown(
                f'<div class="cam-frame"><img src="data:image/jpeg;base64,'
                f'{frame_to_b64(frame)}"/></div>',
                unsafe_allow_html=True,
            )

            st.session_state.current_label = smoothed_label
            st.session_state.current_conf  = confidence
            st.session_state.next_count    = next_count
            st.session_state.prev_count    = prev_count

        cap.release()
        st.session_state.running = False
        st.rerun()