import os
import numpy as np
import gradio as gr
import kagglehub
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# ==============================
# DOWNLOAD DATASET
# ==============================
dataset_path = kagglehub.dataset_download("princelv84/dogsvscats")
print("Dataset path:", dataset_path)
print("Inside dataset:", os.listdir(dataset_path))
DATASET_PATH = dataset_path  
# Find actual train folder
DATASET_PATH = os.path.join(dataset_path, "train")

MODEL_PATH = "model.h5"
CLASS_FILE = "classes.npy"
print("Dataset Path:", DATASET_PATH)

# ==============================
# BUILD MODEL
# ==============================
def build_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ==============================
# TRAIN MODEL
# ==============================
def train_model():
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_data = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(128,128),
        batch_size=32,
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(128,128),
        batch_size=32,
        subset='validation'
    )

    class_names = list(train_data.class_indices.keys())

    # ✅ SAVE CLASS NAMES
    np.save(CLASS_FILE, class_names)

    model = build_model(train_data.num_classes)

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=5
    )
    model.fit(
    train_data,
    validation_data=val_data,
    epochs=3,
    steps_per_epoch=50,        # 🔥 LIMIT TRAINING
    validation_steps=20
)
    model.save(MODEL_PATH)

    return "Training Completed!"

# ==============================
# PREDICT
# ==============================
def predict_image(image):
    model = load_model(MODEL_PATH)

    # ✅ LOAD CLASS NAMES
    class_names = np.load(CLASS_FILE, allow_pickle=True)

  
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    label = class_names[class_index]

    return f"{label} ({confidence:.2f})"
# ==============================
# GRADIO UI
# ==============================
with gr.Blocks() as app:
    gr.Markdown("# 🐶🐱 Cats vs Dogs Classifier")

    train_btn = gr.Button("Train Model")
    train_output = gr.Textbox()

    train_btn.click(train_model, outputs=train_output)

    predict_input = gr.Image(type="numpy")
    predict_btn = gr.Button("Predict")
    predict_output = gr.Textbox()

    predict_btn.click(predict_image, inputs=predict_input, outputs=predict_output)

app.launch(share=True)
