import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import os
from config import DATA_CONFIG

# Image training configuration
IMG_SIZE = (299, 299)  # Xception input size
BATCH_SIZE = 32
EPOCHS = 10

def build_xception_model():
    base_model = Xception(
        include_top=False,
        weights='imagenet',
        input_shape=(*IMG_SIZE, 3),
        pooling='avg'
    )
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model():
    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        preprocessing_function=tf.keras.applications.xception.preprocess_input
    )
    
    train_generator = train_datagen.flow_from_directory(
        DATA_CONFIG['IMAGE_TRAIN_PATH'],
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    # Build and train model
    model = build_xception_model()
    model.fit(
        train_generator,
        epochs=EPOCHS,
        steps_per_epoch=train_generator.samples // BATCH_SIZE
    )
    
    # Save model
    model.save('../backend/models/xception_image.h5')

if __name__ == '__main__':
    train_model()