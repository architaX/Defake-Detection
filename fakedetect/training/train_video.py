import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import cv2
import numpy as np
import os
from config import DATA_CONFIG

# Video training configuration
FRAMES_PER_VIDEO = 30
IMG_SIZE = (299, 299)  # Xception input size
BATCH_SIZE = 8
EPOCHS = 20

def build_video_model():
    # Frame feature extractor (Xception)
    base_model = Xception(
        include_top=False,
        weights='imagenet',
        input_shape=(*IMG_SIZE, 3),
        pooling='avg'
    )
    base_model.trainable = False
    
    frame_input = layers.Input(shape=(*IMG_SIZE, 3))
    frame_features = base_model(frame_input)
    frame_feature_extractor = models.Model(frame_input, frame_features)
    
    # Video model (sequence of frames)
    video_input = layers.Input(shape=(FRAMES_PER_VIDEO, *IMG_SIZE, 3))
    frame_features = layers.TimeDistributed(frame_feature_extractor)(video_input)
    
    # Temporal modeling
    x = layers.LSTM(256)(frame_features)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    return models.Model(video_input, output)

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Get total frames and select evenly spaced indices
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames-1, FRAMES_PER_VIDEO, dtype=int)
    
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, IMG_SIZE)
            frame = tf.keras.applications.xception.preprocess_input(frame)
            frames.append(frame)
    
    cap.release()
    return np.array(frames)

def train_model():
    # Load and preprocess data
    X, y = [], []
    
    for label, class_name in enumerate(['real', 'fake']):
        class_dir = os.path.join(DATA_CONFIG['VIDEO_TRAIN_PATH'], class_name)
        for file in os.listdir(class_dir):
            if file.endswith('.mp4') or file.endswith('.avi'):
                video_path = os.path.join(class_dir, file)
                frames = extract_frames(video_path)
                if len(frames) == FRAMES_PER_VIDEO:
                    X.append(frames)
                    y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    # Build and train model
    model = build_video_model()
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(
        X, y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2
    )
    
    # Save model
    model.save('../backend/models/xception_video.h5')

if __name__ == '__main__':
    train_model()