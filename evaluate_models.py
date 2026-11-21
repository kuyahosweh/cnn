import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

TEST_DIR = "dataset/test"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# IMPORTANT: shuffle=False ensures predictions stay aligned with true labels!
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# Get ground-truth labels (0 for cat, 1 for dog)
true_labels = test_generator.classes
class_names = list(test_generator.class_indices.keys())

model_files = ["simple_cnn.keras", "deeper_cnn.keras", "optimized_cnn.keras"]

for model_file in model_files:
    print(f"\n{'='*50}")
    print(f"EVALUATING: {model_file}")
    print(f"{'='*50}")
    
    model = load_model(model_file)
    
    # Predict probabilities (values between 0.0 and 1.0)
    raw_predictions = model.predict(test_generator)
    
    # Convert probabilities to binary class: > 0.5 becomes 1 (dog), <= 0.5 becomes 0 (cat)
    predicted_labels = (raw_predictions > 0.5).astype(int).flatten()
    
    # Print Precision, Recall, F1-Score, and Accuracy
    print(classification_report(true_labels, predicted_labels, target_names=class_names))
    
    # Generate and save Confusion Matrix image
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_file}")
    plt.savefig(f"cm_{model_file}.png")
    plt.close()
    print(f"Saved confusion matrix chart to cm_{model_file}.png")
