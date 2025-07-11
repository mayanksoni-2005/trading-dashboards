import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Stock Trading Simulator", layout="wide")
st.title("📈 Stock Trading Simulator")

# Strategy Selection
st.markdown("### Step 1: Select Strategy")
col1, col2, col3 = st.columns(3)
strategy = st.session_state.get("strategy", None)

with col1:
    if st.button("SMA Crossover Strategy"):
        strategy = "SMA"
        st.session_state["strategy"] = strategy

with col2:
    if st.button("RSI Strategy"):
        strategy = "RSI"
        st.session_state["strategy"] = strategy

with col3:
    if st.button("Bollinger Bands Strategy"):
        strategy = "BOLL"
        st.session_state["strategy"] = strategy

st.markdown(
    """
    <div style="display: flex; justify-content: space-between; margin-top: -10px;">
        <a href="https://www.investopedia.com/terms/s/sma.asp" target="_blank">📘 SMA Strategy</a>
        <a href="https://www.investopedia.com/terms/r/rsi.asp" target="_blank">📘 RSI Strategy</a>
        <a href="https://www.investopedia.com/terms/b/bollingerbands.asp" target="_blank">📘 Bollinger Bands</a>
    </div>
    """,
    unsafe_allow_html=True
)


if strategy:
    st.info(f"Selected Strategy: **{strategy}**")

    # Step 2: Common Inputs
    st.markdown("### Step 2: Input Stock & Date Range")
    ticker = st.text_input("Enter Stock Ticker", value="AAPL")
    start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))

    strat_inputs = {}

    # Step 3: Strategy-specific parameters
    st.markdown("### Step 3: Strategy Parameters")
    if strategy == "SMA":
        strat_inputs["short_window"] = st.number_input("Short SMA Window", 1, 100, value=20)
        strat_inputs["long_window"] = st.number_input("Long SMA Window", strat_inputs["short_window"] + 1, 200, value=50)
    elif strategy == "RSI":
        strat_inputs["rsi_window"] = st.number_input("RSI Window", 1, 50, value=14)
        strat_inputs["rsi_buy"] = st.slider("Buy when RSI <", 0, 50, value=30)
        strat_inputs["rsi_sell"] = st.slider("Sell when RSI >", 50, 100, value=70)
    elif strategy == "BOLL":
        strat_inputs["boll_window"] = st.number_input("Moving Average Window", 1, 100, value=20)
        strat_inputs["boll_stddev"] = st.slider("Number of Std Dev for Bands", 1.0, 4.0, value=2.0)

    # Step 4: Apply strategy
    if st.button("✅ Apply Strategy"):
        if not ticker:
            st.warning("Please enter a valid ticker symbol.")
        else:
            data = yf.download(ticker, start=start_date, end=end_date)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if data.empty:
                st.warning("No data found for the selected ticker and date range.")
            else:
                st.subheader(f"📉 Historical Data for {ticker}")
                st.dataframe(data)

                initial_cash = 10000
                cash = initial_cash
                shares = 0

                if strategy == "SMA":
                    short_window = strat_inputs["short_window"]
                    long_window = strat_inputs["long_window"]
                    if short_window >= long_window:
                        st.error("❌ Short SMA Window must be **less than** Long SMA Window.")
        

                    data['SMA_Short'] = data['Close'].rolling(window=short_window).mean()
                    data['SMA_Long'] = data['Close'].rolling(window=long_window).mean()
                    data['Signal'] = 0
                    data['Signal'][long_window:] = np.where(
                        data['SMA_Short'][long_window:] > data['SMA_Long'][long_window:], 1, 0
                    )
                    data['Position'] = data['Signal'].diff()

                    st.line_chart(data[['Close', 'SMA_Short', 'SMA_Long']])

                    buy_signals = data[data['Position'] == 1]
                    sell_signals = data[data['Position'] == -1]
                    st.subheader("Buy Signals (SMA Crossover Up)")
                    st.dataframe(buy_signals[['Close', 'SMA_Short', 'SMA_Long']])
                    st.subheader("Sell Signals (SMA Crossover Down)")
                    st.dataframe(sell_signals[['Close', 'SMA_Short', 'SMA_Long']])

                elif strategy == "RSI":
                    rsi_window = strat_inputs["rsi_window"]
                    rsi_buy = strat_inputs["rsi_buy"]
                    rsi_sell = strat_inputs["rsi_sell"]

                    delta = data['Close'].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    avg_gain = gain.rolling(window=rsi_window).mean()
                    avg_loss = loss.rolling(window=rsi_window).mean()
                    rs = avg_gain / avg_loss
                    data['RSI'] = 100 - (100 / (1 + rs))

                    data['Position'] = 0
                    data.loc[data['RSI'] < rsi_buy, 'Position'] = 1
                    data.loc[data['RSI'] > rsi_sell, 'Position'] = -1

                    st.line_chart(data[['Close', 'RSI']])

                    buy_signals = data[data['Position'] == 1]
                    sell_signals = data[data['Position'] == -1]
                    st.subheader("Buy Signals (RSI Oversold)")
                    st.dataframe(buy_signals[['Close', 'RSI']])
                    st.subheader("Sell Signals (RSI Overbought)")
                    st.dataframe(sell_signals[['Close', 'RSI']])
                
                elif strategy == "BOLL":
                    window = strat_inputs["boll_window"]
                    stddev = strat_inputs["boll_stddev"]

                    data['Middle'] = data['Close'].rolling(window=window).mean()
                    data['StdDev'] = data['Close'].rolling(window=window).std()
                    data['Upper'] = data['Middle'] + stddev * data['StdDev']
                    data['Lower'] = data['Middle'] - stddev * data['StdDev']

                    # Buy when price crosses below lower band
                    # Sell when price crosses above upper band
                    data['Position'] = 0
                    data['Position'] = np.where(data['Close'] < data['Lower'], 1, data['Position'])
                    data['Position'] = np.where(data['Close'] > data['Upper'], -1, data['Position'])

                    st.line_chart(data[['Close', 'Middle', 'Upper', 'Lower']])

                    buy_signals = data[data['Position'] == 1]
                    sell_signals = data[data['Position'] == -1]

                    st.subheader("Buy Signals (Below Lower Band)")
                    st.dataframe(buy_signals[['Close', 'Lower']])

                    st.subheader("Sell Signals (Above Upper Band)")
                    st.dataframe(sell_signals[['Close', 'Upper']])

                # Trading Simulation
                for idx, row in data.iterrows():
                    if row['Position'] == 1 and cash > 0:
                        shares = cash // row['Close']
                        cash -= shares * row['Close']
                    elif row['Position'] == -1 and shares > 0:
                        cash += shares * row['Close']
                        shares = 0

                final_value = cash + shares * data['Close'].iloc[-1]
                st.success(f"💰 Final Portfolio Value: ${final_value:,.2f} (Initial: ${initial_cash:,.2f})")
