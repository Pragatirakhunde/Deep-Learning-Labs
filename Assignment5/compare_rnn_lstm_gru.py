import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense, Dropout
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# Configuration
# ============================================================
NUM_WORDS = 10000
MAX_LEN = 200
EMBED_DIM = 64
UNITS = 64
EPOCHS = 5
BATCH_SIZE = 128
SEED = 42

RESULTS_DIR = "results"
MODELS_DIR = "models"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# 1. Load IMDB dataset
# ============================================================
print("Loading IMDB dataset...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=NUM_WORDS)

print(f"Training samples: {len(x_train)}")
print(f"Testing samples : {len(x_test)}")

# ============================================================
# 2. Pad sequences
# ============================================================
x_train = pad_sequences(
    x_train,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

x_test = pad_sequences(
    x_test,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

print("Training shape:", x_train.shape)
print("Testing shape :", x_test.shape)

# ============================================================
# 3. Model builder
# ============================================================
def build_model(model_type):
    model = Sequential([
        Embedding(
            input_dim=NUM_WORDS,
            output_dim=EMBED_DIM,
            input_length=MAX_LEN
        )
    ])

    if model_type == "RNN":
        model.add(SimpleRNN(UNITS))

    elif model_type == "LSTM":
        model.add(LSTM(UNITS))

    elif model_type == "GRU":
        model.add(GRU(UNITS))

    else:
        raise ValueError("model_type must be RNN, LSTM or GRU")

    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ============================================================
# 4. Train and evaluate all three models
# ============================================================
model_types = ["RNN", "LSTM", "GRU"]

histories = {}
results = []
reports = {}

for model_type in model_types:
    print("\n" + "=" * 70)
    print(f"Training {model_type}")
    print("=" * 70)

    tf.keras.backend.clear_session()
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    model = build_model(model_type)
    model.summary()

    start_time = time.time()

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.10,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=1
    )

    training_time = time.time() - start_time

    # Prediction
    probabilities = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
    predictions = (probabilities >= 0.5).astype(int).ravel()

    # Metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    results.append({
        "Model": model_type,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Training_Time_sec": training_time
    })

    histories[model_type] = history.history

    report = classification_report(
        y_test,
        predictions,
        target_names=["Negative", "Positive"],
        digits=4,
        zero_division=0
    )
    reports[model_type] = report

    print(f"\n{model_type} Results")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    print("\nClassification Report:")
    print(report)

    # Save model
    model.save(os.path.join(MODELS_DIR, f"{model_type}.keras"))

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_title(f"{model_type} Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1], ["Negative", "Positive"])
    ax.set_yticks([0, 1], ["Negative", "Positive"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, f"confusion_matrix_{model_type}.png"),
        dpi=150
    )
    plt.close(fig)

# ============================================================
# 5. Save numerical results
# ============================================================
results_df = pd.DataFrame(results)
results_df.to_csv(
    os.path.join(RESULTS_DIR, "metrics.csv"),
    index=False
)

# ============================================================
# 6. Save classification reports
# ============================================================
with open(
    os.path.join(RESULTS_DIR, "classification_reports.txt"),
    "w",
    encoding="utf-8"
) as f:
    for model_type, report in reports.items():
        f.write("=" * 70 + "\n")
        f.write(f"{model_type} Classification Report\n")
        f.write("=" * 70 + "\n")
        f.write(report + "\n\n")

# ============================================================
# 7. Model comparison chart
# ============================================================
metrics = ["Accuracy", "Precision", "Recall", "F1_Score"]
x = np.arange(len(model_types))
width = 0.18

fig, ax = plt.subplots(figsize=(11, 6))

for i, metric in enumerate(metrics):
    values = results_df[metric].values
    ax.bar(x + (i - 1.5) * width, values, width, label=metric)

ax.set_title("RNN vs LSTM vs GRU - Performance Comparison")
ax.set_xlabel("Model")
ax.set_ylabel("Score")
ax.set_xticks(x)
ax.set_xticklabels(model_types)
ax.set_ylim(0, 1.05)
ax.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "model_comparison.png"),
    dpi=150
)
plt.close(fig)

# ============================================================
# 8. Training curves
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6))

for model_type in model_types:
    ax.plot(
        histories[model_type]["accuracy"],
        label=f"{model_type} Train"
    )
    ax.plot(
        histories[model_type]["val_accuracy"],
        linestyle="--",
        label=f"{model_type} Validation"
    )

ax.set_title("Training and Validation Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "training_curves.png"),
    dpi=150
)
plt.close(fig)

# ============================================================
# 9. Print final comparison
# ============================================================
print("\n" + "=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)
print(results_df.to_string(index=False))

best_model = results_df.loc[
    results_df["F1_Score"].idxmax(),
    "Model"
]

print(f"\nBest model according to F1-score: {best_model}")

print("\nFiles generated in the results/ folder:")
for filename in sorted(os.listdir(RESULTS_DIR)):
    print(" -", filename)

print("\nTrained models generated in the models/ folder:")
for filename in sorted(os.listdir(MODELS_DIR)):
    print(" -", filename)

print("\nAssignment implementation completed successfully.")
