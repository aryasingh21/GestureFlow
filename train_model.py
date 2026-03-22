import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

img_size   = 64
batch_size = 32

# Replace your datagen with this:
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,        # randomly rotate images
    width_shift_range=0.1,    # shift left/right
    height_shift_range=0.1,   # shift up/down
    brightness_range=[0.7, 1.3],  # vary brightness
    horizontal_flip=False,    # don't flip — left/right gestures matter
    zoom_range=0.1,           # slight zoom variation
)
train_data = datagen.flow_from_directory(
    'dataset',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    'dataset',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

print("Class indices:", train_data.class_indices)

# FIX 1: Use Input() layer instead of input_shape= argument
# This prevents the 'batch_shape' deserialisation crash when loading the model
model = Sequential([
    Input(shape=(64, 64, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
    
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(train_data, validation_data=val_data, epochs=10)

# FIX 2: Save to ROOT folder — presentation.py looks here, not model/ subfolder
model.save("gesture_model.h5")

print("\nDone! Saved to gesture_model.h5")
print("Class order:", sorted(train_data.class_indices.keys()))

# ── Evaluate on validation set as test set ────────────────────────────────────
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Use val_data as test set
val_data.reset()
y_true, y_pred = [], []

for i in range(len(val_data)):
    x_batch, y_batch = val_data[i]
    preds = model.predict(x_batch, verbose=0)
    y_true.extend(np.argmax(y_batch, axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

class_names = sorted(train_data.class_indices.keys())
print("\n=== TEST RESULTS ===")
print(classification_report(y_true, y_pred, target_names=class_names))