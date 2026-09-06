import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ==============================
# PROJECT PATHS
# ==============================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

train_dir = os.path.join(
    BASE_DIR,
    "dataset",
    "dataset2-master",
    "images",
    "TRAIN"
)

test_dir = os.path.join(
    BASE_DIR,
    "dataset",
    "dataset2-master",
    "images",
    "TEST"
)

# ==============================
# IMAGE SETTINGS
# ==============================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# ==============================
# LOAD TRAINING DATA
# ==============================

train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    shuffle=True
)

# ==============================
# LOAD VALIDATION DATA
# ==============================

validation_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    shuffle=True
)

# ==============================
# LOAD TEST DATA
# ==============================

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==============================
# CLASS INFORMATION
# ==============================

class_names = train_data.class_names

print("\nClasses:", class_names)
print("Image size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)

# ==============================
# PERFORMANCE
# ==============================

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(AUTOTUNE)
validation_data = validation_data.prefetch(AUTOTUNE)
test_data = test_data.prefetch(AUTOTUNE)

# ==============================
# DATA AUGMENTATION
# ==============================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.20),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomBrightness(factor=0.2, value_range=(0, 255)),
    layers.RandomContrast(factor=0.2)
])

# ==============================
# EFFICIENTNETB0
# ==============================

base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers initially
base_model.trainable = False

# ==============================
# BUILD MODEL
# ==============================

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dense(256, activation="relu")(x)

x = layers.Dropout(0.5)(x)

outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = models.Model(inputs, outputs)

# ==============================
# STAGE 1 COMPILE
# ==============================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# ==============================
# MODEL PATHS
# ==============================

stage1_path = os.path.join(BASE_DIR, "model", "best_stage1_model.keras")
stage2_path = os.path.join(BASE_DIR, "model", "best_stage2_model.keras")

os.makedirs(os.path.dirname(stage1_path), exist_ok=True)

# ==============================
# STAGE 1 CALLBACKS
# ==============================

stage1_callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        mode="max",
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=0.00001
    ),
    ModelCheckpoint(
        stage1_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True
    )
]

# ==============================
# STAGE 1
# TRANSFER LEARNING
# ==============================

print("\n==============================")
print("STAGE 1: TRANSFER LEARNING")
print("==============================\n")

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=15,
    callbacks=stage1_callbacks
)

# ==============================
# LOAD BEST STAGE 1 MODEL
# ==============================

print("\nLoading best Stage 1 model...")
model = tf.keras.models.load_model(stage1_path)

# ==============================
# STAGE 2
# FINE-TUNING
# ==============================

print("\n==============================")
print("STAGE 2: FINE-TUNING")
print("==============================\n")

base_model = model.layers[2]
base_model.trainable = True

# Freeze early layers
for layer in base_model.layers[:-40]:
    layer.trainable = False

# Keep BatchNormalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

# ==============================
# RECOMPILE
# ==============================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# ==============================
# STAGE 2 CALLBACKS
# ==============================

stage2_callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        mode="max",
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=0.0000001
    ),
    ModelCheckpoint(
        stage2_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True
    )
]

# ==============================
# FINE-TUNING
# ==============================

fine_tune_history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=15,
    callbacks=stage2_callbacks
)

# ==============================
# LOAD BEST FINE-TUNED MODEL
# ==============================

print("\nLoading best fine-tuned model...")
best_model = tf.keras.models.load_model(stage2_path)

# ==============================
# FINAL TEST EVALUATION
# ==============================

print("\n==============================")
print("FINAL TEST EVALUATION")
print("==============================\n")

test_loss, test_accuracy = best_model.evaluate(test_data)

# ==============================
# FINAL RESULTS
# ==============================

print("\nFinal Test Accuracy:", round(test_accuracy * 100, 2), "%")
print("Final Test Loss:", round(test_loss, 4))
print("\nBest fine-tuned model saved successfully!")
print(stage2_path)