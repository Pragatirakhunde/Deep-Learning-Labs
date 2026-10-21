# Assignment 6 — Tomato Leaf Disease Classification

This practical trains a convolutional neural network (CNN) to classify tomato leaf images by disease category. It includes training visualizations, evaluation metrics, single-image prediction, and a Streamlit interface for using the saved model.

## Files

- `Assignment6.ipynb` — notebook for preparing the image data, training and evaluating the CNN, and saving the model.
- `app.py` — Streamlit application for making predictions with the saved model.
- `tomato_disease_cnn.keras` — saved Keras model.
- `tomato/` — tomato image dataset; the notebook expects the training images under `tomato/train/`.
- `requirement.txt` — Python package dependencies.
- `Assignment6.pdf` — practical report.

## Requirements

Python packages listed in `requirement.txt`: Streamlit, TensorFlow, NumPy, and Pillow.

To launch the prediction app after installing the requirements:

```bash
streamlit run app.py
```
