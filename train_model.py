import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- CONFIGURATION ---
DATA_DIR = 'processed_data'  # This folder must contain 'real' and 'fake' subfolders
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 10  # How many times to study the data

# 1. Load the Data (Images)
# This automatically splits your data: 80% for training, 20% for testing
print("🔄 Loading images...")

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# 2. Build the "MesoNet" Brain
# A specialized network for deepfake detection
print("🧠 Building the model...")

model = Sequential([
    # Layer 1
    Conv2D(8, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    # Layer 2
    Conv2D(8, (5, 5), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    # Layer 3
    Conv2D(16, (5, 5), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    # Flattening (turning the 2D image features into a list of numbers)
    Flatten(),
    
    # Decision Making Layers
    Dense(16, activation='relu'),
    Dropout(0.5), # Prevents memorizing exact images
    Dense(1, activation='sigmoid') # Output: 0 (Real) to 1 (Fake)
])

# 3. Compile the Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4. Train!
print("🚀 Starting training... (This might take a while)")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# 5. Save the Brain
model.save('veriface_model.h5')
print("✅ Model saved as 'veriface_model.h5'. You are ready for the app!")