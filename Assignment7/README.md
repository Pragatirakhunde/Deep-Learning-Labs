# Transfer Learning Assignment - TensorFlow/Keras

## Models
1. AlexNet-style CNN baseline
2. VGG16 - ImageNet pretrained
3. ResNet50 - ImageNet pretrained
4. EfficientNetB0 - ImageNet pretrained

## Dataset
CIFAR-10, restricted to Cat and Dog.
- Training: 500 Cat + 500 Dog = 1,000 images
- Validation: 100 Cat + 100 Dog = 200 images
- Image size: 128x128
- Batch size: 8
- Epochs: 2

## Important AlexNet note
TensorFlow/Keras `tf.keras.applications` does not provide an official
ImageNet-pretrained AlexNet model. Therefore the project does NOT falsely
claim to use pretrained AlexNet weights. It implements an AlexNet-style
baseline and compares it with the three official Keras ImageNet-pretrained
models.

If your teacher strictly requires pretrained AlexNet, use a converted
AlexNet checkpoint from an approved source and adapt the loader separately.

## Installation
Windows:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python transfer_learning_tensorflow.py
```

## Output
The `results` directory contains the final comparison CSV, comparison graph,
individual accuracy plots, confusion matrices and training histories.
