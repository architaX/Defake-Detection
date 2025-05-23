import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import numpy as np
import librosa
import os
from config import DATA_CONFIG

# Audio training configuration
DURATION = 3  # seconds
SR = 16000    # sample rate
N_MELS = 128  # mel bands
BATCH_SIZE = 32
EPOCHS = 10

def build_audio_model():
    # Input shape for mel-spectrograms
    input_shape = (N_MELS, (SR * DURATION) // 128 + 1, 3)
    
    base_model = EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling='avg'
    )
    
    model = models.Sequential([
        base_model,
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SR)
    
    # Pad/trim audio
    if len(y) > SR * DURATION:
        y = y[:SR * DURATION]
    else:
        padding = SR * DURATION - len(y)
        y = np.pad(y, (0, padding), mode='constant')
    
    # Create mel-spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
    mel = librosa.power_to_db(mel, ref=np.max)
    
    # Normalize and create 3-channel input
    mel = (mel - mel.min()) / (mel.max() - mel.min())
    mel_rgb = np.stack([mel]*3, axis=-1)
    
    return mel_rgb

def train_model():
    # Load and preprocess data
    X, y = [], []
    
    for label, class_name in enumerate(['real', 'fake']):
        class_dir = os.path.join(DATA_CONFIG['AUDIO_TRAIN_PATH'], class_name)
        for file in os.listdir(class_dir):
            if file.endswith('.wav') or file.endswith('.mp3'):
                file_path = os.path.join(class_dir, file)
                features = extract_features(file_path)
                X.append(features)
                y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    # Build and train model
    model = build_audio_model()
    model.fit(
        X, y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2
    )
    
    # Save model
    model.save('../backend/models/efficientnet_audio.h5')

if __name__ == '__main__':
    train_model()