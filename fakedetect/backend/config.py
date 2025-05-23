import os

class Config:
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
    
    # Model paths
    XCEPTION_IMAGE_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'xception_image.h5')
    EFFICIENTNET_AUDIO_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'efficientnet_audio.h5')
    XCEPTION_VIDEO_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'xception_video.h5')