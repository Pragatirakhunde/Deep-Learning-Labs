# LSTM Time-Series Forecasting

A simple college-assignment implementation for next-day stock closing-price forecasting using an LSTM.

## Dataset
The script can download historical stock data using `yfinance`. If internet access is unavailable, place a CSV named `stock_data.csv` in the project folder with a `Close` column.

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
python lstm_stock_forecasting.py
```

The script:
1. Loads historical stock data.
2. Uses the Close price.
3. Scales values with MinMaxScaler.
4. Creates 60-day sequences.
5. Trains an LSTM model.
6. Predicts test-set prices.
7. Calculates MAE, RMSE and MAPE.
8. Saves an Actual-vs-Predicted plot as `actual_vs_predicted.png`.
9. Saves the trained model as `lstm_stock_model.keras`.

This is an educational forecasting experiment, not financial advice.
