import os

# Dataset configuration - these paths should point to your actual dataset locations
DATA_CONFIG = {
    'IMAGE_TRAIN_PATH': os.path.join('C:', 'Users', 'archi', 'Documents', 'image_dataset'),
    'AUDIO_TRAIN_PATH': os.path.join('C:', 'Users', 'archi', 'Documents', 'audio_dataset'),
    'VIDEO_TRAIN_PATH': os.path.join('C:', 'Users', 'archi', 'Documents', 'video_dataset')
}

# Training parameters
TRAIN_CONFIG = {
    'IMAGE_INPUT_SIZE': (299, 299),  # Xception input size
    'AUDIO_DURATION': 3,             # seconds
    'AUDIO_SAMPLE_RATE': 16000,
    'VIDEO_FRAMES': 30               # frames per video
}