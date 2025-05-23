import cv2
import numpy as np
import librosa
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

def preprocess_image(filepath, target_size=(299, 299)):  # Xception uses 299x299
    """Preprocess image for Xception model"""
    img = image.load_img(filepath, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = xception_preprocess(img_array)
    return np.expand_dims(img_array, axis=0)

def preprocess_audio(filepath, target_sr=16000, duration=3):
    """Preprocess audio for EfficientNetB0 model"""
    y, sr = librosa.load(filepath, sr=target_sr)
    
    # Pad/trim audio
    if len(y) > target_sr * duration:
        y = y[:target_sr * duration]
    else:
        padding = target_sr * duration - len(y)
        y = np.pad(y, (0, padding), mode='constant')
    
    # Create mel-spectrogram (EfficientNet expects 3 channels)
    mel = librosa.feature.melspectrogram(y=y, sr=target_sr, n_mels=128)
    mel = librosa.power_to_db(mel, ref=np.max)
    
    # Normalize and create 3-channel input
    mel = (mel - mel.min()) / (mel.max() - mel.min())
    mel_rgb = np.stack([mel]*3, axis=-1)  # Convert to 3 channels
    
    return np.expand_dims(mel_rgb, axis=0)

def preprocess_video(filepath, frames=30, target_size=(299, 299)):
    """Extract and preprocess frames from video for Xception model"""
    cap = cv2.VideoCapture(filepath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames-1, frames, dtype=int)
    
    processed_frames = []
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, target_size)
            frame = xception_preprocess(frame)
            processed_frames.append(frame)
    
    cap.release()
    return np.expand_dims(np.array(processed_frames), axis=0)