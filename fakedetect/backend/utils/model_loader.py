from tensorflow.keras.models import load_model
from config import Config

def load_models():
    """Load all pretrained models into memory"""
    models = {}
    
    try:
        # Load image model (Xception based)
        models['image'] = load_model(Config.XCEPTION_IMAGE_MODEL_PATH)
        
        # Load audio model (EfficientNetB0 based)
        models['audio'] = load_model(Config.EFFICIENTNET_AUDIO_MODEL_PATH)
        
        # Load video model (Xception + LSTM)
        models['video'] = load_model(Config.XCEPTION_VIDEO_MODEL_PATH)
        
        print("All models loaded successfully")
        return models
    
    except Exception as e:
        print(f"Error loading models: {e}")
        raise e