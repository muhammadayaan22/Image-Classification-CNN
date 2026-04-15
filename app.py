import os
import numpy as np
import cv2
import streamlit as st
import kagglehub
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Cats vs Dogs", layout="centered")

MODEL_PATH = "model.h5"
CLASS_FILE = "classes.npy"

# ==============================
# DOWNLOAD DATASET
# ==============================
@st.cache_resource
def load_dataset():
    dataset_path = kagglehub.dataset_download("princelv84/dogsvscats")
    dataset_path = os.path.join(dataset_path, "train")
    return dataset_path

DATASET_PATH = load_dataset()

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

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
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
    np.save(CLASS_FILE, class_names)

    model = build_model(train_data.num_classes)

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=3,
        steps_per_epoch=50,
        validation_steps=20
    )

    model.save(MODEL_PATH)

    return "✅ Training Completed!"

# ==============================
# PREDICT
# ==============================
def predict_image(uploaded_file):
    model = load_model(MODEL_PATH)
    class_names = np.load(CLASS_FILE, allow_pickle=True)

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    img = cv2.resize(img, (128,128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    return f"{class_names[class_index]} ({confidence:.2f})"

# ==============================
# UI
# ==============================
st.title("🐶🐱 Cats vs Dogs Classifier")

# ---- TRAIN ----
if st.button("Train Model"):
    with st.spinner("Training... please wait ⏳"):
        result = train_model()
    st.success(result)

# ---- PREDICT ----
st.subheader("Upload Image for Prediction")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        result = predict_image(uploaded_file)
        st.success(f"Prediction: {result}")