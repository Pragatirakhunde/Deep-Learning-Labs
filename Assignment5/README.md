# Assignment 2: RNN vs LSTM vs GRU for Sequence Classification

## Problem Statement
Implement and compare RNN, LSTM, and GRU models for sequence classification and analyze their performance using appropriate evaluation metrics.

## Dataset
This implementation uses the IMDB movie-review sentiment dataset available through TensorFlow/Keras.
The task is binary sequence classification:
- 0 = Negative
- 1 = Positive

The dataset is automatically downloaded by Keras on the first run.

## Requirements
Python 3.10+ is recommended.

Install:
```bash
pip install -r requirements.txt
```

Run:
```bash
python compare_rnn_lstm_gru.py
```

## What the program does
1. Loads the IMDB dataset.
2. Keeps the top 10,000 words.
3. Pads reviews to 200 tokens.
4. Builds three comparable models:
   - SimpleRNN
   - LSTM
   - GRU
5. Uses the same Embedding, hidden-unit count, optimizer, epochs and batch size for fair comparison.
6. Trains and evaluates each model.
7. Calculates:
   - Accuracy
   - Precision
   - Recall
   - F1-score
8. Generates confusion matrices.
9. Generates a model-performance comparison chart.
10. Generates training/validation accuracy and loss plots.
11. Saves all results to `results/`.
12. Saves trained models to `models/`.

## Output files
- `results/metrics.csv`
- `results/model_comparison.png`
- `results/training_curves.png`
- `results/confusion_matrix_RNN.png`
- `results/confusion_matrix_LSTM.png`
- `results/confusion_matrix_GRU.png`
- `results/classification_reports.txt`
- `models/RNN.keras`
- `models/LSTM.keras`
- `models/GRU.keras`

## Notes
Training three neural networks can take several minutes depending on CPU/GPU.
The script is intended for educational comparison, not for claiming that one architecture is universally superior.
