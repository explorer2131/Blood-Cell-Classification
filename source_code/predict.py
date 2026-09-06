import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

# ==============================
# PROJECT PATH
# ==============================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

model_path = os.path.join(
    BASE_DIR,
    "model",
    "best_stage2_model.keras"
)

# ==============================
# SETTINGS
# ==============================

IMG_SIZE = (224, 224)

# ==============================
# LOAD MODEL
# ==============================

print("Loading trained model...")

model = tf.keras.models.load_model(model_path)

print("Model loaded successfully!")

# Extract exact class names from the saved model to avoid indexing bugs
if hasattr(model, 'class_names'):
    class_names = model.class_names
else:
    # Fallback to standard dataset order if not embedded in metadata
    class_names = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]

# ==============================
# IMAGE PATH
# ==============================

image_path = input(
    "\nEnter the path of the blood cell image: "
).strip()

# ==============================
# CHECK IMAGE
# ==============================

if not os.path.exists(image_path):
    print("\nError: Image file not found.")
    exit()

# ==============================
# LOAD AND PREPROCESS IMAGE
# ==============================

image = load_img(
    image_path,
    target_size=IMG_SIZE
)

image_array = img_to_array(image)

image_array = np.expand_dims(
    image_array,
    axis=0
)

# ==============================
# PREDICTION
# ==============================

print("\nPredicting...")

predictions = model.predict(
    image_array,
    verbose=0
)

predicted_index = np.argmax(
    predictions[0]
)

predicted_class = class_names[
    predicted_index
]

confidence = predictions[0][
    predicted_index
] * 100

# ==============================
# RESULT
# ==============================

print("\n==============================")
print("       PREDICTION RESULT")
print("==============================")

print("Predicted Cell :", predicted_class)
print("Confidence     :", f"{confidence:.2f}%")

print("==============================")