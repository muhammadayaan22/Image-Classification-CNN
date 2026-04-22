import numpy as np
import cv2
import gradio as gr
from tensorflow.keras.models import load_model

# ==============================
# LOAD MODEL + CLASSES
# ==============================
MODEL_PATH = "model.h5"
CLASS_FILE = "classes.npy"

model = load_model(MODEL_PATH)
class_names = np.load(CLASS_FILE, allow_pickle=True)

# ==============================
# PREDICT FUNCTION
# ==============================
def predict_image(image):
    if image is None:
        return "Please upload an image."

    # Resize + normalize
    img = cv2.resize(image, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)[0]

    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    return f"🐾 Prediction: {class_names[class_index]} ({confidence:.2f})"

# ==============================
# UI
# ==============================
with gr.Blocks() as demo:
    gr.Markdown("# 🐶🐱 Cats vs Dogs Classifier")
    gr.Markdown("Upload an image and the model will predict whether it's a cat or a dog.")

    image_input = gr.Image(type="numpy", label="Upload Image")
    predict_btn = gr.Button("Predict")

    output = gr.Textbox(label="Result")

    predict_btn.click(fn=predict_image, inputs=image_input, outputs=output)

# ==============================
# LAUNCH
# ==============================
demo.launch()