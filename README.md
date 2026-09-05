# Blood Cell Classification using Deep Learning

## Project Overview

Blood Cell Classification is a deep learning based image classification project that identifies different types of white blood cells from microscopic blood cell images.

The system uses EfficientNetB0 with transfer learning and fine-tuning to classify images into four categories:

- Eosinophil
- Lymphocyte
- Monocyte
- Neutrophil

A Flask-based web application allows users to upload an image, view the predicted class and confidence score, and check previous prediction history using SQLite.

## Dataset

The project uses the **Blood Cell Images Dataset (dataset2-master)**, containing microscopic images of four types of white blood cells:

- Eosinophil
- Lymphocyte
- Monocyte
- Neutrophil

The dataset is organized into separate training and testing folders.

The dataset is not included in this repository because of its large size.

## Model

The classification model is based on **EfficientNetB0** using transfer learning and fine-tuning.

The training process includes:

- Image preprocessing
- Data augmentation
- Transfer learning
- Fine-tuning
- Model evaluation

The final model achieved approximately **69.56% test accuracy**.

## Technologies Used

- Python
- TensorFlow
- Keras
- EfficientNetB0
- Flask
- HTML
- CSS
- SQLite

## Features

- Blood cell image classification
- Confidence score
- Web-based graphical user interface
- Prediction result page
- Previous prediction history
- SQLite database

## Blood Cell Classes

| Class | Description |
|---|---|
| Eosinophil | Type of white blood cell |
| Lymphocyte | Type of white blood cell |
| Monocyte | Type of white blood cell |
| Neutrophil | Type of white blood cell |

## Project Structure

```text
Blood-Cell-Classification/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── model/
│   ├── best_stage1_model.keras
│   └── best_stage2_model.keras
│
├── source_code/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── check_dataset.py
│
├── templates/
│   ├── index.html
│   ├── Result.html
│   └── history.html
│
└── static/
    └── uploads/
        └── .gitkeep
```

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Flask application:

```bash
python app.py
```

Then open the local address displayed in the terminal in a web browser.

Upload a blood cell image to view its predicted class and confidence score.

## Prediction History

Previous prediction details are stored in a local SQLite database. The database is created automatically when the application is run and is not included in this repository.

## Important Notes

- The dataset is not included because of its large size.
- The local SQLite database is not included.
- User-uploaded images are not stored in the repository.
- The trained model files are included for prediction.

## Future Scope

- Improve model accuracy
- Increase dataset diversity
- Experiment with other deep learning architectures
- Deploy the application online
- Add more blood cell categories

## Project Type

**Machine Learning / Deep Learning / Computer Vision**
