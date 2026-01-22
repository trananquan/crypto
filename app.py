import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Add custom CSS for the button
st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #003366; /* Dark blue background */
        color: white; /* White text */
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }
    div.stButton > button:hover {
        background-color: #002244; /* Slightly darker blue on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page setup
st.markdown(
    """
    <h1 style='color: darkblue;'>📊 Phân tích và Dự báo giá Crypto bằng thuật toán AI- Máy học</h1>
    """,
    unsafe_allow_html=True
)

# User inputs
symbol = st.selectbox("Chọn mã Crypto (vd: BTC-USD)",
                    options=["BTC-USD", "ETH-USD", "USDT-USD", "XRP-USD", "BNB-USD", "SOL-USD", "USDC-USD", "DOGE-USD", "TRX-USD", "ADA-USD"],
                    index=0  # Default selection is BTC-USD
)

start_date = st.date_input("Ngày đầu", value=date(2024, 1, 1))
end_date = st.date_input("Ngày cuối", value=date.today())

if symbol:
    # Fetch cryptocurrency data
    crypto_data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    crypto_data.reset_index(inplace=True)

    st.write(f"Dữ liệu giá tiền điện tử {symbol} từ ngày {start_date} đến ngày {end_date}:")
    st.dataframe(crypto_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']])
    crypto_data.rename(columns={"Date": "time", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)


# Checkboxes for indicators
st.subheader("📈 Chỉ báo kỹ thuật")
col1, col2 = st.columns(2)  # Create two columns
with col1:
     sma20 = st.checkbox("Đường SMA 20")
     sma50 = st.checkbox("Đường SMA 50")
     macd_checkbox = st.checkbox("Chỉ báo MACD")
with col2:
     ema20 = st.checkbox("Đường EMA 20")
     bbands = st.checkbox("Dải Bollinger Bands")
     rsi_checkbox = st.checkbox("Chỉ báo RSI")
# Button to show chart
if st.button("Xem biểu đồ"):
    # Download data
    data = yf.download(symbol, start=start_date, end=end_date)
    
    if data.empty:
        st.error("No data found for this symbol and date range.")
    else:
        # Calculate indicators
        if sma20:
            data['SMA20'] = data['Close'].rolling(window=20).mean()
        if sma50:
            data['SMA50'] = data['Close'].rolling(window=50).mean()
        if ema20:
            data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
        if bbands:
           close_price = data['Close']
           if isinstance(close_price, pd.DataFrame):
              close_price = close_price.iloc[:, 0]  # get first column if multiple
    
           data['BB_Middle'] = close_price.rolling(window=20).mean()
           bb_std = close_price.rolling(window=20).std()
           data['BB_Upper'] = data['BB_Middle'] + (2 * bb_std)
           data['BB_Lower'] = data['BB_Middle'] - (2 * bb_std)
        if macd_checkbox:
            ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = ema_12 - ema_26
            data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        if rsi_checkbox:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))


  
        # Plotting
        plt.figure(figsize=(12,6))
        plt.plot(data['Close'], label='Close Price', color='black')

        if sma20:
            plt.plot(data['SMA20'], label='SMA 20', color='blue')
        if sma50:
            plt.plot(data['SMA50'], label='SMA 50', color='purple')
        if ema20:
            plt.plot(data['EMA20'], label='EMA 20', color='orange')
        if bbands:
            plt.plot(data['BB_Upper'], label='Bollinger Upper', color='red', linestyle='--')
            plt.plot(data['BB_Lower'], label='Bollinger Lower', color='red', linestyle='--')
            plt.fill_between(data.index, data['BB_Lower'], data['BB_Upper'], color='lightcoral', alpha=0.2)

        plt.title(f"{symbol} Biểu đồ giá Crypto với các đường chỉ báo")
        plt.xlabel("Ngày")
        plt.ylabel("Giá (USD)")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

        # Show MACD Chart separately
        if macd_checkbox:
            fig_macd, ax_macd = plt.subplots(figsize=(12, 3))
            ax_macd.plot(data['MACD'], label='MACD', color='blue')
            ax_macd.plot(data['Signal'], label='Signal', color='orange')
            ax_macd.set_title('MACD')
            ax_macd.legend()
            ax_macd.grid(True)
            st.pyplot(fig_macd)

        # Show RSI Chart separately
        if rsi_checkbox:
            fig_rsi, ax_rsi = plt.subplots(figsize=(12, 3))
            ax_rsi.plot(data['RSI'], label='RSI', color='purple')
            ax_rsi.axhline(70, color='red', linestyle='--')
            ax_rsi.axhline(30, color='green', linestyle='--')
            ax_rsi.set_title('RSI')
            ax_rsi.legend()
            ax_rsi.grid(True)
            st.pyplot(fig_rsi)


def summarize_timeline_and_recommend(data, symbol):
    """
    Summarize the last 30 days of stock data and get a buy/sell recommendation from Gemini AI.
    """
    # Filter the last 30 rows
    last_30_days = data.tail(30)

    # Calculate summary statistics
    avg_price = float(last_30_days['Close'].mean())
    max_price = float(last_30_days['Close'].max())
    min_price = float(last_30_days['Close'].min())
    latest_price = float(last_30_days['Close'].iloc[-1])

    # Create a summary dictionary
    summary = {
        "Average Price": avg_price,
        "Highest Price": max_price,
        "Lowest Price": min_price,
        "Latest Price": latest_price,
    }

    # Get a recommendation from Gemini AI
    recommendation = get_gemini_recommendation(symbol, summary)
    summary["Recommendation"] = recommendation

    return summary

# Checkbox for prediction models
st.write()
st.subheader("📈 Mô hình dự đoán giá Crypto")

# Model checkboxes
prophet_checkbox = st.checkbox("Mô hình Prophet")
random_forest_checkbox = st.checkbox("Random Forest")
gradient_boosting_checkbox = st.checkbox("Gradient Boosting")

# Button to predict
if st.button("Dự đoán giá Crypto"):
    
    data = yf.download(symbol, start=start_date, end=end_date)

    if data.empty:
        st.error("Không có dữ liệu cho mã Crypto.")
    else:
        predictions = {}

        # Prepare data for models
        data['Date'] = data.index
        data.reset_index(drop=True, inplace=True)
        data['ds'] = data['Date']
        data['y'] = data['Close']
                
        # Prophet model
        if prophet_checkbox:
            try:
                prophet_model = Prophet()
                prophet_model.fit(data[['ds', 'y']])
                future = prophet_model.make_future_dataframe(periods=30)
                forecast = prophet_model.predict(future)
                predictions['Prophet'] = forecast['yhat'][-30:].values
            except Exception as e:
                st.error(f"Prophet error: {e}")

        # Create lag features for Random Forest & Gradient Boosting
        n_lags = 5
        for i in range(1, n_lags+1):
            data[f'lag_{i}'] = data['Close'].shift(i)
        data.dropna(inplace=True)

        X = data[[f'lag_{i}' for i in range(1, n_lags+1)]]
        y = data['Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Random Forest model
        if random_forest_checkbox:
            try:
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)

                last_known = X.iloc[-1].values.reshape(1, -1)
                rf_forecast = []

                for _ in range(30):
                    pred = rf_model.predict(last_known)[0]
                    rf_forecast.append(pred)

                    # Update lag values
                    last_known = np.roll(last_known, shift=-1)
                    last_known[0, -1] = pred

                predictions['Random Forest'] = rf_forecast

            except Exception as e:
                st.error(f"Random Forest error: {e}")

        # Gradient Boosting model
        if gradient_boosting_checkbox:
            try:
                gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                gb_model.fit(X_train, y_train)

                last_known = X.iloc[-1].values.reshape(1, -1)
                gb_forecast = []

                for _ in range(30):
                    pred = gb_model.predict(last_known)[0]
                    gb_forecast.append(pred)

                    # Update lag values
                    last_known = np.roll(last_known, shift=-1)
                    last_known[0, -1] = pred

                predictions['Gradient Boosting'] = gb_forecast

            except Exception as e:
                st.error(f"Gradient Boosting error: {e}")

        # Display predictions
        if predictions:
            prediction_df = pd.DataFrame(predictions)
            prediction_df.index = pd.date_range(start=end_date + pd.Timedelta(days=1), periods=30)
            prediction_df.index.name = "Date"

            prediction_df['Giá trị trung bình'] = prediction_df.mean(axis=1)

            st.write("Dự đoán giá trong 30 ngày tới:")
            st.dataframe(prediction_df)

            # Plot the predictions
            plt.figure(figsize=(12, 6))
            for column in prediction_df.columns:
                plt.plot(prediction_df.index, prediction_df[column], label=column)

            plt.title(f"Dự đoán giá Crypto trong 30 ngày tới ({symbol})")
            plt.xlabel("Ngày")
            plt.ylabel("Giá (USD)")
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)

