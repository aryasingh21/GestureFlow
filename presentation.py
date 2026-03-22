"""
presentation.py  —  GestureFlow gesture daemon (CNN Image Model Edition)
Run this DIRECTLY from terminal: python presentation.py
Do NOT rely on Streamlit to launch it — cv2.imshow needs its own terminal.
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import os
import sys
import time
import collections

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[gesture] ERROR: pip install tensorflow")

_SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH       = os.path.join(_SCRIPT_DIR, "gesture_model.h5")
DATASET_DIR      = os.path.join(_SCRIPT_DIR, "dataset")
EXIT_SIGNAL_FILE = os.path.join(_SCRIPT_DIR, ".gesture_exit")
IMG_SIZE         = 64
CONFIDENCE_THRESHOLD = 0.60
SMOOTHING_WINDOW     = 4
SWIPE_COOLDOWN       = 0.8
ZOOM_COOLDOWN        = 0.6
EXIT_HOLD_SEC        = 1.2
pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0.0

def get_class_names():
    if not os.path.isdir(DATASET_DIR):
        return sorted(["exit", "freeze", "next", "palm", "prev", "zoom_in", "zoom_out"])
    classes = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    print(f"[gesture] Class order: {classes}")
    return classes

def load_model():
    if not TF_AVAILABLE:
        return None
    if not os.path.exists(MODEL_PATH):
        print(f"[gesture] ERROR: Model not found at {MODEL_PATH}")
        return None
    print(f"[gesture] Loading: {MODEL_PATH}")
    try:
        m = keras.models.load_model(MODEL_PATH)
        print(f"[gesture] OK — input={m.input_shape}  output={m.output_shape}")
        return m
    except Exception as e:
        print(f"[gesture] Load failed: {e}")
        return None

def get_hand_crop(frame, hand_landmarks, padding=0.25):
    h, w = frame.shape[:2]
    lms  = hand_landmarks.landmark
    xs   = [lm.x for lm in lms]
    ys   = [lm.y for lm in lms]
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
    crop = crop.astype(np.float32) / 255.0
    return crop.reshape(1, IMG_SIZE, IMG_SIZE, 3)

def write_exit_signal():
    try:
        with open(EXIT_SIGNAL_FILE, "w") as f:
            f.write("exit")
        print("[gesture] Exit signal written.")
    except Exception as e:
        print(f"[gesture] Could not write exit signal: {e}")

def smooth_prediction(pred_buffer, new_pred):
    pred_buffer.append(new_pred)
    return max(set(pred_buffer), key=pred_buffer.count)

def draw_hud(frame, label, colour, freeze_mode, raw_label, confidence, smoothed):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, "GestureFlow [CNN]", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    if label:
        cv2.putText(frame, label, (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 1.4, colour, 3, cv2.LINE_AA)
    if freeze_mode:
        cv2.putText(frame, "FROZEN — show palm to unlock", (w//2 - 180, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (80, 80, 255), 2, cv2.LINE_AA)
    px = w - 270
    cv2.rectangle(frame, (px - 6, 4), (w - 4, 115), (15, 15, 15), -1)
    cv2.rectangle(frame, (px - 6, 4), (w - 4, 115), (60, 60, 60), 1)
    cv2.putText(frame, f"RAW    : {raw_label}", (px, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 60), 1)
    cv2.putText(frame, f"SMOOTH : {smoothed}", (px, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 220, 80), 1)
    bar_col = (0, 200, 80) if confidence >= CONFIDENCE_THRESHOLD else (0, 80, 220)
    cv2.rectangle(frame, (px, 54), (px + 230, 66), (40, 40, 40), -1)
    cv2.rectangle(frame, (px, 54), (px + int(230 * min(confidence, 1.0)), 66), bar_col, -1)
    cv2.putText(frame, f"CONF   : {int(confidence*100)}%  (min={int(CONFIDENCE_THRESHOLD*100)}%)", (px, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.38, bar_col, 1)
    cv2.putText(frame, "palm=Ready  freeze=Lock  exit(hold)=Quit  next/prev=Slide", (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 120, 120), 1)

def main():
    print("=" * 60)
    print("  GestureFlow — CNN Gesture Daemon")
    print("=" * 60)

    class_names = get_class_names()
    num_classes = len(class_names)
    model = load_model()
    if model is None:
        print("[gesture] Cannot run without model. Exiting.")
        write_exit_signal()
        input("Press Enter to close...")
        return

    model.predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32), verbose=0)
    print("[gesture] Model warm-up done.")

    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils

    cap = None
    for idx in [0, 1, 2]:
        print(f"[gesture] Trying camera index {idx}...")
        _cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if _cap.isOpened():
            ret, test_frame = _cap.read()
            if ret and test_frame is not None:
                cap = _cap
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                print(f"[gesture] Camera {idx} confirmed working!")
                break
            else:
                print(f"[gesture] Camera {idx} opened but cannot read frames.")
                _cap.release()
        else:
            print(f"[gesture] Camera {idx} failed to open.")
            _cap.release()

    if cap is None:
        print("[gesture] ERROR: No working camera found.")
        write_exit_signal()
        input("Press Enter to close...")
        return

    WINDOW = "GestureFlow"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 800, 600)
    print(f"[gesture] Window ready. Starting loop — press Q or ESC to quit.")

    pred_buffer      = collections.deque(maxlen=SMOOTHING_WINDOW)
    exit_start_time  = None
    last_swipe_time  = 0.0
    last_zoom_time   = 0.0
    freeze_mode      = False
    ready_for_next   = True
    last_acted_label = None
    frame_count      = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.60,
        min_tracking_confidence=0.50,
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[gesture] Frame read failed — retrying...")
                time.sleep(0.05)
                continue

            frame_count += 1
            if frame_count % 60 == 0:
                print(f"[gesture] Running... frame {frame_count}")

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = hands.process(rgb)
            now   = time.time()

            label          = None
            colour         = (255, 255, 255)
            raw_label      = "no_hand"
            smoothed_label = "no_hand"
            confidence     = 0.0

            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                crop = get_hand_crop(frame, hand)
                if crop is not None:
                    probs      = model.predict(crop, verbose=0)[0]
                    top_idx    = int(np.argmax(probs))
                    confidence = float(probs[top_idx])
                    raw_label  = (
                        class_names[top_idx]
                        if confidence >= CONFIDENCE_THRESHOLD and top_idx < len(class_names)
                        else "low_conf"
                    )
                else:
                    raw_label = "low_conf"

                smoothed_label = smooth_prediction(pred_buffer, raw_label)

                if raw_label == "exit":
                    if exit_start_time is None:
                        exit_start_time = now
                    held     = now - exit_start_time
                    progress = min(held / EXIT_HOLD_SEC, 1.0)
                    hf, wf   = frame.shape[:2]
                    cv2.rectangle(frame, (0, hf - 12), (int(wf * progress), hf), (0, 0, 220), -1)
                    cv2.putText(frame, f"Hold to Exit  {int(progress*100)}%", (10, hf - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 255), 2)
                    if held >= EXIT_HOLD_SEC:
                        print("[gesture] Exit gesture confirmed.")
                        cap.release()
                        cv2.destroyAllWindows()
                        write_exit_signal()
                        return
                    draw_hud(frame, label, colour, freeze_mode, raw_label, confidence, smoothed_label)
                    cv2.imshow(WINDOW, frame)
                    cv2.waitKey(1)
                    continue
                else:
                    exit_start_time = None

                if smoothed_label == "freeze":
                    freeze_mode = True; ready_for_next = False; last_acted_label = "freeze"
                    label, colour = "FROZEN", (60, 60, 255)

                elif smoothed_label in ("palm", "open_palm"):
                    freeze_mode = False; ready_for_next = True; last_acted_label = smoothed_label
                    label, colour = "READY", (80, 220, 80)

                elif freeze_mode:
                    label, colour = "Show PALM to unlock", (0, 200, 220)

                elif not ready_for_next:
                    label, colour = "Show PALM to continue", (180, 180, 0)

                elif smoothed_label in ("next", "swipe_left") and smoothed_label != last_acted_label:
                    if (now - last_swipe_time) >= SWIPE_COOLDOWN:
                        pyautogui.press("right")
                        print("[gesture] ACTION: NEXT SLIDE")
                        label, colour = ">> NEXT SLIDE", (80, 220, 80)
                        last_swipe_time = now; ready_for_next = False; freeze_mode = True; last_acted_label = smoothed_label

                elif smoothed_label in ("prev", "swipe_right") and smoothed_label != last_acted_label:
                    if (now - last_swipe_time) >= SWIPE_COOLDOWN:
                        pyautogui.press("left")
                        print("[gesture] ACTION: PREV SLIDE")
                        label, colour = "<< PREV SLIDE", (220, 120, 0)
                        last_swipe_time = now; ready_for_next = False; freeze_mode = True; last_acted_label = smoothed_label

                elif smoothed_label == "zoom_in" and smoothed_label != last_acted_label:
                    if (now - last_zoom_time) >= ZOOM_COOLDOWN:
                        pyautogui.hotkey("ctrl", "+")
                        print("[gesture] ACTION: ZOOM IN")
                        label, colour = "ZOOM IN  +", (80, 220, 120)
                        last_zoom_time = now; ready_for_next = False; freeze_mode = True; last_acted_label = "zoom_in"

                elif smoothed_label == "zoom_out" and smoothed_label != last_acted_label:
                    if (now - last_zoom_time) >= ZOOM_COOLDOWN:
                        pyautogui.hotkey("ctrl", "-")
                        print("[gesture] ACTION: ZOOM OUT")
                        label, colour = "ZOOM OUT  -", (80, 120, 220)
                        last_zoom_time = now; ready_for_next = False; freeze_mode = True; last_acted_label = "zoom_out"

                else:
                    label = f"Waiting... {int(confidence*100)}%"
                    colour = (120, 120, 120)

            else:
                pred_buffer.clear()
                exit_start_time  = None
                last_acted_label = None

            draw_hud(frame, label, colour, freeze_mode, raw_label, confidence, smoothed_label)
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                print("[gesture] Quit key pressed.")
                break

    cap.release()
    cv2.destroyAllWindows()
    write_exit_signal()
    print("[gesture] Done.")

if __name__ == "__main__":
    main()