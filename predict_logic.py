import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

def create_dummy_model():
    """Create a dummy LSTM model for demonstration purposes."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(100, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def load_or_create_model():
    """Load the trained model or create a dummy one if file doesn't exist."""
    try:
        from tensorflow.keras.models import load_model
        if os.path.exists('keras_model.h5'):
            model = load_model('keras_model.h5')
            print("✅ Loaded existing model from keras_model.h5")
        else:
            model = create_dummy_model()
            print("⚠️ Model file not found, using dummy model for demonstration")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return create_dummy_model()

def fetch_data():
    """Download historical AAPL stock data from Yahoo Finance."""
    try:
        # Set an end date to today to ensure full range
        end_date = datetime.today().strftime('%Y-%m-%d')
        data = yf.download('AAPL', start='2010-01-01', end=end_date, progress=False, auto_adjust=False)

        # Ensure 'Close' column exists and drop NaNs
        if 'Close' not in data.columns or data.empty:
            raise ValueError("Downloaded data is empty or missing 'Close' column.")

        close_data = data[['Close']].dropna().reset_index(drop=True)

        # Optional debug line
        print(f"✅ Fetched {len(close_data)} records from Yahoo Finance.")

        return close_data, data
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        # Return dummy data if fetch fails
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        dummy_prices = 150 + np.cumsum(np.random.randn(len(dates)) * 2)
        dummy_data = pd.DataFrame({
            'Close': dummy_prices,
            'High': dummy_prices + np.random.rand(len(dates)) * 5,
            'Low': dummy_prices - np.random.rand(len(dates)) * 5,
            'Open': dummy_prices + np.random.randn(len(dates)) * 2,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        close_data = dummy_data[['Close']].reset_index(drop=True)
        return close_data, dummy_data

def preprocess(data):
    """Scale and reshape the last 100 closing prices for prediction."""
    if len(data) < 100:
        raise ValueError(f"❌ Not enough data to make a prediction. Only {len(data)} records found.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    last_100 = scaled_data[-100:]
    input_data = np.array([last_100])
    input_data = input_data.reshape((1, 100, 1))

    return input_data, scaler

def get_trading_signal(current_price, predicted_price):
    """Generate trading signal based on price prediction."""
    price_change = predicted_price - current_price
    price_change_pct = (price_change / current_price) * 100
    
    if price_change_pct > 2:
        return "Buy", "🟢"
    elif price_change_pct < -2:
        return "Sell", "🔴"
    else:
        return "Hold", "🟡"

def predict():
    """Make a prediction using the loaded LSTM model."""
    try:
        model = load_or_create_model()
        data, full_data = fetch_data()
        input_data, scaler = preprocess(data)

        # Make future prediction
        pred_scaled = model.predict(input_data, verbose=0)
        pred_actual = scaler.inverse_transform(pred_scaled)

        # Calculate metrics using historical data
        mae, mse, r2 = calculate_prediction_metrics(model, data, scaler)

        return float(pred_actual[0][0]), full_data, mae, mse, r2
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return 200.0, None, None, None, None


def get_recent_data():
    """Get the most recent 5 rows of AAPL stock data."""
    try:
        ticker = yf.Ticker("AAPL")
        # Get recent data
        hist = ticker.history(period="5d", interval="1d")
        
        if hist.empty:
            # Fallback to dummy data
            dates = pd.date_range(end=datetime.now(), periods=5, freq='D')
            hist['Dividends'] = 0
            hist['Stock Splits'] = 0
            hist = pd.DataFrame({
                'Open': [200, 202, 198, 205, 203],
                'High': [205, 207, 203, 210, 208],
                'Low': [198, 200, 195, 203, 201],
                'Close': [202, 201, 199, 207, 205],
                'Volume': [50000000, 48000000, 52000000, 55000000, 49000000]
            }, index=dates)
        
        # Format the data
        hist = hist.round(2)
        hist['Volume'] = hist['Volume'].astype(int)
        
        return hist.tail(5)
    except Exception as e:
        print(f"❌ Error getting recent data: {e}")
        # Return dummy data
        dates = pd.date_range(end=datetime.now(), periods=5, freq='D')
        return pd.DataFrame({
            'Open': [200, 202, 198, 205, 203],
            'High': [205, 207, 203, 210, 208],
            'Low': [198, 200, 195, 203, 201],
            'Close': [202, 201, 199, 207, 205],
            'Volume': [50000000, 48000000, 52000000, 55000000, 49000000]
        }, index=dates)
def calculate_prediction_metrics(model, data, scaler):
    """Calculate prediction metrics (MAE, MSE, R2) using historical data."""
    sequence_length = 100

    if len(data) <= sequence_length:
        return None, None, None

    scaled_data = scaler.fit_transform(data)

    X = []
    y = []

    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    predictions_scaled = model.predict(X, verbose=0)
    predictions = scaler.inverse_transform(predictions_scaled)
    actual = scaler.inverse_transform(y.reshape(-1, 1))

    # Calculate metrics
    mae = mean_absolute_error(actual, predictions)
    mse = mean_squared_error(actual, predictions)
    r2 = r2_score(actual, predictions)

    return mae, mse, r2
