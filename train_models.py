import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================================
# 1. DEFINE PATHS AND HYPERPARAMETERS
# ==========================================
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20  # Increase to 15-20 for higher accuracy once you verify it runs

# Verify folders exist before starting
for path in [TRAIN_DIR, TEST_DIR]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory not found: {path}. Check your folder setup.")

# ==========================================
# 2. DATA PREPROCESSING & GENERATORS
# ==========================================
# Rescale pixel values from [0, 255] to [0, 1] and reserve 20% for validation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

# Test data should ONLY be rescaled (never split or augmented)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

print("\nLoading Training Data...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

print("\nLoading Validation Data...")
validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

print("\nLoading Test Data...")
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# Print class mapping (usually {'cat': 0, 'dog': 1})
print("\nClass Indices Mapping:", train_generator.class_indices)

# ==========================================
# 3. MODEL 1: SIMPLE CNN (1 Conv Block)
# ==========================================
def build_simple_cnn():
    model = models.Sequential([
        # Input: 128x128 RGB image (3 color channels: Red, Green, Blue)
        layers.Input(shape=(128, 128, 3)),
        
        # 32 filters of size 3x3 scan the image for basic edges/colors
        layers.Conv2D(32, (3, 3), activation="relu"),
        # MaxPooling cuts width & height in half (126x126 -> 63x63) to keep strongest features
        layers.MaxPooling2D((2, 2)),
        
        # Flatten converts the 2D feature maps into a 1D list of numbers
        layers.Flatten(),
        # Standard Dense neural network layer
        layers.Dense(64, activation="relu"),
        # Sigmoid outputs a single probability between 0.0 (cat) and 1.0 (dog)
        layers.Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ==========================================
# 4. MODEL 2: DEEPER CNN (3 Conv Blocks)
# ==========================================
def build_deeper_cnn():
    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        
        # Block 1: Extracts basic edges (32 filters)
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2: Combines edges into textures/shapes (64 filters)
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3: Recognizes high-level parts like ears/snouts (128 filters)
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ==========================================
# 5. MODEL 3: OPTIMIZED CNN (BatchNorm + Dropout)
# ==========================================
def build_optimized_cnn():
    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.BatchNormalization(),  # Stabilizes and speeds up learning
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),         # Randomly turns off 25% of neurons to prevent memorization
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),          # Stronger dropout before final decision
        layers.Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ==========================================
# 6. TRAIN AND SAVE ALL THREE MODELS
# ==========================================
architectures = {
    "simple_cnn.keras": build_simple_cnn(),
    "deeper_cnn.keras": build_deeper_cnn(),
    "optimized_cnn.keras": build_optimized_cnn()
}

for filename, model in architectures.items():
    print(f"\n{'='*50}")
    print(f"TRAINING: {filename}")
    print(f"{'='*50}")
    model.summary()
    
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )
    
    model.save(filename)
    print(f"Saved model successfully to {filename}")
