import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from predict_logic import predict, get_recent_data, get_trading_signal
from urllib.parse import urlparse, parse_qs
import streamlit.web.server.websocket_headers as wh

st.set_page_config(page_title="AAPL Stock App", layout="wide")

# --- Navigation State Initialization ---
if "nav" not in st.session_state:
    st.session_state.nav = "Home"

# ✅ Get ?nav=... from URL manually (HTML form buttons rely on this)
try:
    headers = st.context.headers()
    url = headers.get("Referer", "")
    query_params = parse_qs(urlparse(url).query)
    if "nav" in query_params:
        st.session_state.nav = query_params["nav"][0]
except Exception:
    pass  # Ignore if headers are unavailable

# --- Custom CSS ---
st.markdown("""
    <style>
    body, .stApp {
        background: linear-gradient(135deg, #000000 0%, #0d1b2a 30%, #1b263b 60%, #431010 100%);
        color: #ffffff;
        font-family: 'Courier New', monospace;
    }

    .header-box {
        background-color: #0d1b2a;
        padding: 1rem 2rem;
        border-bottom: 2px solid #00d4ff;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
    }

    .nav-button {
        background-color: transparent;
        color: #00d4ff;
        font-size: 1rem;
        font-weight: bold;
        border: none;
        margin: 0 1rem;
        cursor: pointer;
        text-decoration: none;
    }

    .nav-button:hover {
        text-decoration: underline;
    }

    .section-title {
        color: #00d4ff;
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a2332 0%, #0f1419 100%);
        border: 1px solid #00d4ff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.1);
    }

    .metric-title {
        color: #00d4ff;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    .metric-value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .metric-change {
        font-size: 0.9rem;
        color: #8892b0;
    }
    </style>
""", unsafe_allow_html=True)
# --- HEADER + NAVBAR (Styled Streamlit Buttons) ---
st.markdown("""
    <style>
    .nav-container {
        display: flex;
        justify-content: flex-start;
        gap: 1rem;
        margin-top: 1rem;
    }
    .stButton>button {
        background-color: transparent;
        border: none;
        color: #00d4ff;
        font-size: 1rem;
        font-weight: bold;
        cursor: pointer;
        transition: 0.3s;
    }
    .stButton>button:hover {
        text-decoration: underline;
        color: #00ffff;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header-box">
        <h1 style="color:white; margin:0;">AAPL Price Predictor</h1>
        <div class="nav-container">
    """, unsafe_allow_html=True)

# Navigation Buttons (trigger Streamlit rerun correctly)
nav1, nav2, nav3, nav4 = st.columns([1, 1, 1, 1])
with nav1:
    if st.button("Home"):
        st.session_state.nav = "Home"
with nav2:
    if st.button("Data"):
        st.session_state.nav = "Data"
with nav3:
    if st.button("Prediction"):
        st.session_state.nav = "Prediction"
with nav4:
    if st.button("Metrics"):
        st.session_state.nav = "Metrics"

st.markdown("</div></div>", unsafe_allow_html=True)


# --- HOME PAGE ---
if st.session_state.nav == "Home":
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; height: 50vh; flex-direction: column; text-align: center;">
        <h1 style="color: #00d4ff; font-size: 3rem; margin-bottom: 1rem;">Welcome!!</h1>
        <p style="color: #ffffffcc; font-size: 1.2rem; max-width: 700px;">
            This AI-powered web app uses a deep learning model (LSTM) to predict future prices of Apple Inc. (AAPL) stock. 
            Navigate using the menu above to explore real-time stock data, view predictions, assess model performance metrics, 
            and receive a trading signal based on the latest AI forecast.
        </p>
    </div>
""", unsafe_allow_html=True)


# --- DATA PAGE ---
elif st.session_state.nav == "Data":
    st.markdown('<h2 class="section-title">📊 Recent Stock Data</h2>', unsafe_allow_html=True)
    try:
        df = get_recent_data()
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading data: {e}")

# --- PREDICTION PAGE ---
elif st.session_state.nav == "Prediction":
    st.markdown('<h2 class="section-title">🔮 AI Price Prediction</h2>', unsafe_allow_html=True)
    if st.button("🚀 Run LSTM Prediction"):
        with st.spinner("Running prediction..."):
            predicted_price, full_data, mae, mse, r2 = predict()
            df = get_recent_data()
            current_price = df['Close'].iloc[-1]
            signal, emoji = get_trading_signal(current_price, predicted_price)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Current Price</div>
                <div class="metric-value">${current_price:.2f}</div>
                <div class="metric-title" style="margin-top:1rem;">Predicted</div>
                <div class="metric-value" style="color: #00d4ff;">${predicted_price:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Trading Signal</div>
                <div class="metric-value">{emoji} {signal}</div>
                <div class="metric-change">Expected Δ: ${predicted_price - current_price:+.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<h3 class="section-title">📈 Price Chart</h3>', unsafe_allow_html=True)
        try:
            history = full_data['Close'].tail(30).astype(float).values.flatten()
            fig, ax = plt.subplots(figsize=(6, 3), facecolor='white')

            # Plot the historical prices
            ax.plot(history, color='#0077cc', label="Historical")

            # Plot the prediction line
            ax.plot(
                [len(history) - 1, len(history)],
                [history[-1], predicted_price],
                'r--o',
                label="Prediction"
            )

            # --- Styling ---
            ax.set_facecolor('black')  # White background inside plot
            ax.set_xlabel("Time", color='black', fontsize=10)  # X-axis label
            ax.set_ylabel("Price", color='black', fontsize=10)  # Y-axis label
            ax.set_title("AAPL Price Prediction", color='black', fontsize=12)

            ax.tick_params(colors='black')  # Black ticks
            ax.grid(True, linestyle='--', color='gray', linewidth=0.5)  # Grid
            ax.legend()

            # Display plot
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Chart error: {e}")

# --- METRICS PAGE ---
elif st.session_state.nav == "Metrics":
    st.markdown('<h2 class="section-title">📈 Model Performance Metrics</h2>', unsafe_allow_html=True)
    # --- In Prediction tab only ---
    predicted_price, full_data, mae, mse, r2 = predict()
# then plot charts and display metrics

    if None in (mae, mse, r2):
        st.error("Could not calculate metrics. Try rerunning the prediction.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">R² Score</div>
                <div class="metric-value" style="color: #00ff88;">{r2:.4f}</div>
                <div class="metric-change">Fit quality</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">MSE</div>
                <div class="metric-value" style="color: #00ff88;">{mse:.2f}</div>
                <div class="metric-change">Mean Squared Error</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">MAE</div>
                <div class="metric-value" style="color: #ffa502;">{mae:.2f}</div>
                <div class="metric-change">Mean Absolute Error</div>
            </div>
            """, unsafe_allow_html=True)
