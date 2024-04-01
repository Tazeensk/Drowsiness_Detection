import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Initialize image data generators with rescaling and data augmentation
train_data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
validation_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess train and validation images
train_generator = train_data_gen.flow_from_directory(
    'data/train',  # path to training data directory
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='binary'  # binary classification: open or closed
)

validation_generator = validation_data_gen.flow_from_directory(
    'data/validation',  # path to validation data directory
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='binary'
)

# Define model architecture
eye_model = Sequential()
eye_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
eye_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
eye_model.add(MaxPooling2D(pool_size=(2, 2)))
eye_model.add(Dropout(0.25))

eye_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
eye_model.add(MaxPooling2D(pool_size=(2, 2)))
eye_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
eye_model.add(MaxPooling2D(pool_size=(2, 2)))
eye_model.add(Dropout(0.25))

eye_model.add(Flatten())
eye_model.add(Dense(1024, activation='relu'))
eye_model.add(Dropout(0.5))
eye_model.add(Dense(1, activation='sigmoid'))  # binary classification

# Compile model
eye_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

# Train the model
history = eye_model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# Save model structure to JSON
model_json = eye_model.to_json()
with open("eye_state_model.json", "w") as json_file:
    json_file.write(model_json)

# Save trained model weights
eye_model.save_weights('eye_state_model_weights.h5')
