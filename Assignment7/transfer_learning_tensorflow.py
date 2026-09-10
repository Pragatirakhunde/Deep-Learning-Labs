"""
Deep Learning Assignment
Transfer Learning using VGG16, ResNet50, EfficientNetB0 and AlexNet-style CNN
Framework: TensorFlow / Keras

IMPORTANT:
TensorFlow/Keras does not provide an official ImageNet-pretrained AlexNet model
in tf.keras.applications. Therefore, the first three models use official
ImageNet-pretrained weights. The AlexNet-style network is included as a
lightweight baseline. This is stated clearly in the report rather than
claiming that non-existent pretrained AlexNet weights were used.

Low-memory setup:
- CIFAR-10 Cat vs Dog
- 500 training images per class = 1,000 images
- 200 validation images
- 128x128 images
- batch size 8
- 2 epochs
- models processed one at a time
"""

import os
import gc
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 2
NUM_PER_CLASS = 500
AUTOTUNE = tf.data.AUTOTUNE

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

print("TensorFlow version:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

# ------------------------------------------------------------
# Dataset: CIFAR-10, Cat (3) and Dog (5)
# ------------------------------------------------------------
(x_train_all, y_train_all), (x_test_all, y_test_all) = keras.datasets.cifar10.load_data()
y_train_all = y_train_all.flatten()
y_test_all = y_test_all.flatten()

def select_cat_dog(x, y, n_each=500):
    cat_idx = np.where(y == 3)[0][:n_each]
    dog_idx = np.where(y == 5)[0][:n_each]
    idx = np.concatenate([cat_idx, dog_idx])
    np.random.shuffle(idx)
    x_sel = x[idx]
    # cat -> 0, dog -> 1
    y_sel = (y[idx] == 5).astype(np.float32)
    return x_sel, y_sel

x_train, y_train = select_cat_dog(x_train_all, y_train_all, NUM_PER_CLASS)
x_val, y_val = select_cat_dog(x_test_all, y_test_all, 100)

print("Training images:", len(x_train))
print("Validation images:", len(x_val))

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(1000, seed=SEED).map(preprocess, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# ------------------------------------------------------------
# Model builders
# ------------------------------------------------------------
def build_vgg16():
    base = VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.RandomFlip("horizontal")(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="VGG16_TransferLearning")

def build_resnet50():
    base = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.RandomFlip("horizontal")(inputs)
    # ResNet50 preprocessing
    x = keras.applications.resnet50.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="ResNet50_TransferLearning")

def build_efficientnet():
    # EfficientNetB0 includes its input rescaling in the Keras implementation.
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.RandomFlip("horizontal")(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="EfficientNetB0_TransferLearning")

def build_alexnet_style():
    """
    Lightweight AlexNet-style architecture.
    There is no official ImageNet-pretrained AlexNet in tf.keras.applications,
    so this is used as a baseline rather than falsely labeling it pretrained.
    """
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.Rescaling(1./255)(x)

    x = layers.Conv2D(64, 11, strides=4, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = layers.Conv2D(192, 5, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = layers.Conv2D(384, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="AlexNet_Style_Baseline")

def run_model(name, builder):
    print("\n" + "=" * 60)
    print("Running:", name)

    model = builder()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1
    )

    probs = model.predict(val_ds, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_val, preds),
        "Precision": precision_score(y_val, preds, zero_division=0),
        "Recall": recall_score(y_val, preds, zero_division=0),
        "F1-Score": f1_score(y_val, preds, zero_division=0)
    }

    print("\nFinal metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\nClassification Report:")
    print(classification_report(y_val, preds, target_names=["Cat", "Dog"], zero_division=0))

    cm = confusion_matrix(y_val, preds)
    np.savetxt(
        os.path.join(RESULT_DIR, f"{name}_confusion_matrix.csv"),
        cm, fmt="%d", delimiter=","
    )

    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0, "Epoch", np.arange(1, len(hist_df) + 1))
    hist_df.to_csv(os.path.join(RESULT_DIR, f"{name}_history.csv"), index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(hist_df["Epoch"], hist_df["accuracy"], marker="o", label="Train Accuracy")
    plt.plot(hist_df["Epoch"], hist_df["val_accuracy"], marker="o", label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{name} Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"{name}_accuracy.png"), dpi=150)
    plt.close()

    del model
    keras.backend.clear_session()
    gc.collect()

    return metrics

models_to_run = [
    ("AlexNet", build_alexnet_style),
    ("VGG16", build_vgg16),
    ("ResNet50", build_resnet50),
    ("EfficientNetB0", build_efficientnet),
]

results = []
for name, builder in models_to_run:
    results.append(run_model(name, builder))

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(RESULT_DIR, "model_comparison.csv"), index=False)

plt.figure(figsize=(9, 5))
x = np.arange(len(results_df))
width = 0.18
for i, metric in enumerate(["Accuracy", "Precision", "Recall", "F1-Score"]):
    plt.bar(x + (i - 1.5) * width, results_df[metric], width, label=metric)
plt.xticks(x, results_df["Model"], rotation=20)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Transfer Learning / CNN Model Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "model_comparison.png"), dpi=150)
plt.close()

print("\n" + "=" * 60)
print("FINAL COMPARISON")
print(results_df.to_string(index=False))
print("\nResults saved in:", RESULT_DIR)
