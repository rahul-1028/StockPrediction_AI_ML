import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Step 1: Fetch Stock Data with Error Handling
def get_stock_data(ticker):
    try:
        data = yf.download(ticker, start='2020-01-01', end='2025-01-01', progress=False)
        if data.empty:
            raise ValueError("No data found for the given ticker.")
        return data
    except Exception as e:
        st.error(f"Error fetching data for ticker {ticker}: {e}")
        return None

# Step 2: Data Preprocessing
def preprocess_data(data):
    try:
        data['50_MA'] = data['Close'].rolling(window=50).mean()
        data.dropna(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error in data preprocessing: {e}")
        return None

# Step 3: Feature Engineering
def create_features(data):
    try:
        data['Daily Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Close'].rolling(window=50).std()
        data.dropna(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error in feature engineering: {e}")
        return None

# Step 4: Model Training - Linear Regression
def train_linear_regression(data):
    try:
        X = data[['50_MA', 'Volatility']]
        y = data['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return model, mse, r2
    except Exception as e:
        st.error(f"Error in model training: {e}")
        return None, None, None

# Step 5: Deep Learning Model - LSTM (Optional for Future)
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 6: Streamlit Web App
def run_streamlit():
    st.title('Stock Price Prediction')

    # Input for stock ticker
    ticker = st.text_input('Enter Stock Ticker (e.g., AAPL, TSLA)', 'AAPL')

    # Fetch and display stock data
    data = get_stock_data(ticker)
    if data is not None:
        st.subheader(f"Data for {ticker}")
        st.dataframe(data.tail())

        # Preprocess data
        data = preprocess_data(data)
        if data is not None:
            st.subheader("Stock Closing Prices")
            st.line_chart(data['Close'])

            # Feature engineering
            data = create_features(data)
            if data is not None:
                # Train linear regression model
                model, mse, r2 = train_linear_regression(data)
                if model is not None:
                    st.subheader("Model Evaluation")
                    st.write(f"Mean Squared Error: {mse}")
                    st.write(f"R2 Score: {r2}")

if __name__ == '__main__':
    run_streamlit()