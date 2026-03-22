"""
dataset_capture.py  —  GestureFlow (3 gestures only)
=====================================================
Gestures: next, prev, exit
Saves 64x64 hand crops (exactly what the model sees at inference)
"""

import cv2
import mediapipe as mp
import os
import time
import random
import numpy as np

LABELS           = ['exit', 'next', 'prev']   # alphabetical = model class order
SAMPLES_PER_GESTURE = 300
PADDING          = 0.30
IMG_SIZE         = 64

for label in LABELS:
    os.makedirs(f'dataset/{label}', exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.55,
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n" + "="*50)
print("  GestureFlow — Dataset Collector (3 gestures)")
print("="*50)
print("\n  1 → exit   (pinch: thumb + index touching)")
print("  2 → next   (index finger pointing RIGHT →)")
print("  3 → prev   (index finger pointing LEFT  ←)")
print("\n  r → retake current gesture")
print("  q → quit")
print("\nTIPS:")
print("  • Make gestures very distinct from each other")
print("  • Slightly vary hand angle/distance during capture")
print("  • Same lighting you'll use when presenting")
print("="*50 + "\n")

current_label  = None
capturing      = False
capture_count  = 0
last_save_time = 0
SAVE_INTERVAL  = 0.033

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result  = hands.process(rgb)
    h, w    = frame.shape[:2]
    display = frame.copy()
    crop_preview = None
    hand_found   = False

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(display, hand, mp_hands.HAND_CONNECTIONS)

        lms   = hand.landmark
        xs    = [lm.x for lm in lms]
        ys    = [lm.y for lm in lms]
        x_min = max(0.0, min(xs) - PADDING)
        y_min = max(0.0, min(ys) - PADDING)
        x_max = min(1.0, max(xs) + PADDING)
        y_max = min(1.0, max(ys) + PADDING)
        px1, py1 = int(x_min * w), int(y_min * h)
        px2, py2 = int(x_max * w), int(y_max * h)

        cv2.rectangle(display, (px1, py1), (px2, py2), (0, 255, 255), 2)

        if px2 > px1 and py2 > py1:
            crop = frame[py1:py2, px1:px2]
            if crop.size > 0:
                crop_preview = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                hand_found   = True

    # Save crop if capturing
    now = time.time()
    if capturing and hand_found and (now - last_save_time) >= SAVE_INTERVAL:
        existing      = len(os.listdir(f'dataset/{current_label}'))
        path          = f'dataset/{current_label}/{existing}.jpg'
        cv2.imwrite(path, crop_preview)
        capture_count += 1
        last_save_time = now

        if capture_count >= SAMPLES_PER_GESTURE:
            capturing = False
            total     = len(os.listdir(f'dataset/{current_label}'))
            print(f"  ✓ Done! '{current_label}' now has {total} photos.")

    # ── Preview box (top-left) ────────────────────────────────────────────────
    if crop_preview is not None:
        preview = cv2.resize(crop_preview, (128, 128))
        display[10:138, 10:138] = preview
        cv2.rectangle(display, (9,9), (139,139), (0,255,255), 1)
        cv2.putText(display, "Model sees this", (10, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,255,255), 1)

    # ── Gesture menu (top-right) ──────────────────────────────────────────────
    mx = w - 260
    cv2.rectangle(display, (mx-6, 4), (w-4, 115), (20,20,20), -1)
    cv2.rectangle(display, (mx-6, 4), (w-4, 115), (60,60,60), 1)
    cv2.putText(display, "PRESS KEY TO CAPTURE", (mx, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,160), 1)
    gestures_display = [
        ("1", "exit",  "pinch"),
        ("2", "next",  "point RIGHT →"),
        ("3", "prev",  "point LEFT ←"),
    ]
    for i, (key, label, hint) in enumerate(gestures_display):
        count = len(os.listdir(f'dataset/{label}'))
        col   = (80,220,80) if count >= SAMPLES_PER_GESTURE else (180,180,180)
        tick  = "✓" if count >= SAMPLES_PER_GESTURE else " "
        active = "►" if label == current_label and capturing else " "
        cv2.putText(display, f"{active}{tick} {key}:{label:<6} {hint} ({count})",
                    (mx, 40 + i*26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1)
    cv2.putText(display, "r=retake  q=quit",
                (mx, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120,120,120), 1)

    # ── Status bar (bottom) ───────────────────────────────────────────────────
    if capturing:
        progress = capture_count / SAMPLES_PER_GESTURE
        bar_w    = int((w - 20) * progress)
        cv2.rectangle(display, (10, h-28), (w-10, h-10), (40,40,40), -1)
        cv2.rectangle(display, (10, h-28), (10+bar_w, h-10), (0,200,80), -1)
        cv2.putText(display,
                    f"Capturing '{current_label}'  {capture_count}/{SAMPLES_PER_GESTURE}",
                    (10, h-34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,80), 2)
        if not hand_found:
            cv2.putText(display, "No hand detected — show your hand!",
                        (10, h-48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,80,255), 2)
    else:
        msg = "Press 1/2/3 to start capturing"
        if current_label:
            done = len(os.listdir(f'dataset/{current_label}'))
            msg  = f"'{current_label}' done ({done} photos) — press next key or q"
        cv2.putText(display, msg,
                    (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140,140,140), 1)

    cv2.imshow("GestureFlow — Dataset Collector", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('r') and current_label:
        folder  = f'dataset/{current_label}'
        deleted = 0
        for f_name in os.listdir(folder):
            os.remove(os.path.join(folder, f_name))
            deleted += 1
        print(f"  Retaking '{current_label}' — deleted {deleted} photos.")
        capturing     = True
        capture_count = 0
        last_save_time = 0

    elif key == ord('1'):
        current_label = 'exit'
        capturing = True; capture_count = 0; last_save_time = 0
        print(f"\nCapturing 'exit' — show PINCH gesture (thumb + index touching)")

    elif key == ord('2'):
        current_label = 'next'
        capturing = True; capture_count = 0; last_save_time = 0
        print(f"\nCapturing 'next' — point index finger clearly to the RIGHT →")

    elif key == ord('3'):
        current_label = 'prev'
        capturing = True; capture_count = 0; last_save_time = 0
        print(f"\nCapturing 'prev' — point index finger clearly to the LEFT ←")

cap.release()
cv2.destroyAllWindows()
hands.close()

print("\n" + "="*50)
print("  Final dataset summary:")
for label in LABELS:
    count  = len(os.listdir(f'dataset/{label}'))
    status = "✓ GOOD" if count >= SAMPLES_PER_GESTURE else f"⚠ NEED {SAMPLES_PER_GESTURE-count} MORE"
    print(f"  {label:<8} {count:>4} photos  {status}")
print("="*50)
print("\nNext: python train_model.py")