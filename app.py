import os
import sqlite3
from datetime import datetime
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.utils import load_img, img_to_array

app = Flask(__name__)

# ==============================
# PROJECT PATHS
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "best_stage2_model.keras"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

# CHANGED: New database name to force clear any bad old rows completely
DB_PATH = os.path.join(
    BASE_DIR,
    "fresh_database.db"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==============================
# DATABASE SETUP
# ==============================

def init_db():
    """Initializes a brand new clean SQLite database with exact 5 columns."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            cell_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Initialize the new clean DB table layout immediately
init_db()

# ==============================
# LOAD MODEL
# ==============================

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# Extract exact class names from the saved model to avoid indexing bugs
if hasattr(model, 'class_names'):
    CLASS_NAMES = model.class_names
else:
    # Safe fallback to standard alphabetical order if not embedded
    CLASS_NAMES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]

# ==============================
# ROUTES
# ==============================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image selected."

    file = request.files["image"]

    if file.filename == "":
        return "No image selected."

    filename = secure_filename(file.filename)

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(image_path)

    # Load and preprocess image
    image = load_img(
        image_path,
        target_size=(224, 224)
    )

    image_array = img_to_array(image)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = np.argmax(
        predictions
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(predictions[0]
        [predicted_index
    ] * 100)

    # Save to SQLite Database with guaranteed 5 full fields
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO predictions (filename, cell_type, confidence, timestamp) VALUES (?, ?, ?, ?)",
            (filename, predicted_class, round(confidence, 2), current_time)
        )
        conn.commit()
        conn.close()
        print(f"Successfully saved entry to DB: {predicted_class}")
    except Exception as e:
        print(f"Database Write Error Log: {e}")

    return render_template(
        "result.html",
        prediction=predicted_class,
        confidence=f"{confidence:.2f}",
        image=filename
    )

@app.route("/history")
def history():
    """Fetches all records from the new clean database safely."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, cell_type, confidence, timestamp FROM predictions ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return render_template("history.html", records=rows)

# ==============================
# RUN APP
# ==============================

if __name__ == "__main__":
    app.run(debug=True)