import os

# Project ka main folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset ka path
dataset_path = os.path.join(
    BASE_DIR,
    "dataset",
    "dataset2-master",
    "images"
)

train_path = os.path.join(dataset_path, "TRAIN")
test_path = os.path.join(dataset_path, "TEST")

classes = [
    "EOSINOPHIL",
    "LYMPHOCYTE",
    "MONOCYTE",
    "NEUTROPHIL"
]

print("=== BLOOD CELL DATASET CHECK ===\n")

for class_name in classes:

    train_folder = os.path.join(train_path, class_name)
    test_folder = os.path.join(test_path, class_name)

    train_count = len(os.listdir(train_folder))
    test_count = len(os.listdir(test_folder))

    print(f"{class_name}:")
    print(f"  TRAIN images: {train_count}")
    print(f"  TEST images:  {test_count}")
    print()