import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2

# Exact path from your sidebar
IMG_DIR = r'dataset\train\pair_of_10\pair_of_10\images'
LBL_DIR = r'dataset\train\pair_of_10\pair_of_10\labels'

def load_data():
    images, labels = [], []
    for img_name in os.listdir(IMG_DIR):
        if img_name.endswith('.jpg'):
            img = cv2.imread(os.path.join(IMG_DIR, img_name))
            images.append(cv2.resize(img, (64, 64)))
            # If label file exists and is NOT empty, it is MARKED (1)
            lbl_path = os.path.join(LBL_DIR, img_name.replace('.jpg', '.txt'))
            labels.append(1 if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0 else 0)
    return np.array(images) / 255.0, np.array(labels)

X, y = load_data()
model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10)
model.save('models/bubble_model.keras')