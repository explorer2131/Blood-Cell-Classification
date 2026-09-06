import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

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

test_dir = os.path.join(
    BASE_DIR,
    "dataset",
    "dataset2-master",
    "images",
    "TEST"
)

# ==============================
# SETTINGS
# ==============================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==============================
# LOAD MODEL
# ==============================

print("Loading trained model...")

model = tf.keras.models.load_model(model_path)

print("Model loaded successfully!")

# ==============================
# LOAD TEST DATA
# ==============================

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_data.class_names

print("\nClasses:", class_names)

# ==============================
# MAKE PREDICTIONS
# ==============================

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in test_data:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ==============================
# CLASSIFICATION REPORT
# ==============================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================\n")

print(cm)

# ==============================
# DISPLAY CONFUSION MATRIX
# ==============================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    xticks_rotation=45
)

plt.title("Blood Cell Classification - Confusion Matrix")

plt.tight_layout()

plt.show()