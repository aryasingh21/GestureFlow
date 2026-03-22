"""
augment_dataset.py  —  GestureFlow Dataset Augmentor
=====================================================
Takes your existing dataset photos and generates augmented
variations until each gesture folder has 500 images.

Augmentations applied (all subtle — preserves gesture shape):
  - Random rotation ±15 degrees
  - Brightness variation ±30%
  - Slight zoom in/out
  - Small horizontal/vertical shifts
  - Gaussian noise

Run: python augment_dataset.py
Then: python train_model.py
"""

import cv2
import numpy as np
import os
import random

DATASET_DIR     = "dataset"
TARGET_PER_CLASS = 500
IMG_SIZE        = 64

LABELS = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])

print("=" * 50)
print("  GestureFlow — Dataset Augmentor")
print("=" * 50)

def augment_image(img):
    """Apply random augmentation to a 64x64 image."""
    h, w = img.shape[:2]

    # 1. Random rotation ±15 degrees
    angle  = random.uniform(-15, 15)
    M      = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    img    = cv2.warpAffine(img, M, (w, h),
                             borderMode=cv2.BORDER_REFLECT)

    # 2. Random brightness ±30%
    factor = random.uniform(0.70, 1.30)
    img    = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # 3. Random zoom (crop + resize back)
    zoom = random.uniform(0.85, 1.0)
    zh   = int(h * zoom)
    zw   = int(w * zoom)
    top  = random.randint(0, h - zh)
    left = random.randint(0, w - zw)
    img  = img[top:top+zh, left:left+zw]
    img  = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # 4. Random shift (±4 pixels)
    tx  = random.randint(-4, 4)
    ty  = random.randint(-4, 4)
    M2  = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M2, (w, h),
                          borderMode=cv2.BORDER_REFLECT)

    # 5. Random Gaussian noise (subtle)
    if random.random() > 0.5:
        noise = np.random.normal(0, random.uniform(2, 8),
                                  img.shape).astype(np.int16)
        img   = np.clip(img.astype(np.int16) + noise,
                        0, 255).astype(np.uint8)

    # 6. Random horizontal flip ONLY for palm/freeze/exit
    # (NOT for next/prev/zoom — direction matters for those)
    return img


def should_flip(label):
    """Only flip gestures where left/right doesn't matter."""
    return label in ("palm", "freeze", "exit")


for label in LABELS:
    folder   = os.path.join(DATASET_DIR, label)
    existing = [f for f in os.listdir(folder)
                if f.lower().endswith(('.jpg','.jpeg','.png'))]
    current  = len(existing)

    if current >= TARGET_PER_CLASS:
        print(f"  {label:<12} already has {current} images — skipping")
        continue

    needed = TARGET_PER_CLASS - current
    print(f"  {label:<12} has {current} — generating {needed} more...", end="", flush=True)

    # Load all existing images
    source_imgs = []
    for fname in existing:
        path = os.path.join(folder, fname)
        img  = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            source_imgs.append(img)

    if not source_imgs:
        print(f" ERROR: no readable images found")
        continue

    generated = 0
    while generated < needed:
        # Pick a random source image
        src = random.choice(source_imgs).copy()

        # Apply augmentation
        aug = augment_image(src)

        # Optional flip for symmetric gestures
        if should_flip(label) and random.random() > 0.5:
            aug = cv2.flip(aug, 1)

        # Save with new index
        idx      = current + generated
        out_path = os.path.join(folder, f"aug_{idx}.jpg")
        cv2.imwrite(out_path, aug)
        generated += 1

    final = len(os.listdir(folder))
    print(f" done → {final} total")

print("\n" + "=" * 50)
print("  Final counts:")
total = 0
for label in LABELS:
    count = len([f for f in os.listdir(os.path.join(DATASET_DIR, label))
                 if f.lower().endswith(('.jpg','.jpeg','.png'))])
    total += count
    status = "✓" if count >= TARGET_PER_CLASS else "⚠"
    print(f"  {status} {label:<12} {count} images")
print(f"\n  Total: {total} images across {len(LABELS)} classes")
print("=" * 50)
print("\nNext step: python train_model.py")