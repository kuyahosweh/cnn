import numpy as np
import gradio as gr
from PIL import Image
from tensorflow.keras.models import load_model

# Load your best trained model once when the app starts
MODEL_PATH = "deeper_cnn.keras"
model = load_model(MODEL_PATH)

def predict_image(uploaded_image):
    if uploaded_image is None:
        return "Please upload an image."
    
    # 1. Ensure image is in RGB format (removes alpha channel if user uploads a PNG)
    img = uploaded_image.convert("RGB")
    
    # 2. Resize to the exact 128x128 dimensions the CNN expects
    img = img.resize((128, 128))
    
    # 3. Convert to NumPy array and rescale pixels from [0, 255] to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # 4. Add batch dimension: changes shape from (128, 128, 3) to (1, 128, 128, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    
    # 5. Run prediction (returns probability of class 1 -> 'dog')
    dog_prob = float(model.predict(img_batch)[0][0])
    cat_prob = 1.0 - dog_prob
    
    # Return dictionary of confidence scores for Gradio's Label component
    return {
        "Dog 🐶": dog_prob,
        "Cat 🐱": cat_prob
    }

# Build the web interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload a Cat or Dog Photo"),
    outputs=gr.Label(num_top_classes=2, label="CNN Prediction"),
    title="Cat vs. Dog CNN Image Classifier",
    description="Upload any photo of a cat or dog. The image is resized to 128x128, rescaled to [0,1], and classified by your trained Deeper CNN."
)

if __name__ == "__main__":
    interface.launch()
