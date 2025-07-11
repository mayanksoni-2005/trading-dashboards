import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Backtester:
    def __init__(self, data, signals, initial_capital=100000, transaction_cost=0.001):
        self.data = data.copy()
        self.signals = signals
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

        self.positions = None
        self.trades = None
        self.portfolio = None

    def generate_positions(self):
        self.data['Signal'] = self.signals
        self.data['Position'] = self.data['Signal'].replace(0, method='ffill').fillna(0)
        self.data['Shares'] = 0.0

        # Calculate shares only where Position == 1
        buy_mask = self.data['Position'] == 1
        self.data.loc[buy_mask, 'Shares'] = self.initial_capital / self.data.loc[buy_mask, 'Close']

        # Forward-fill shares during holding period
        self.data['Shares'] = self.data['Shares'].replace(to_replace=0, method='ffill').fillna(0)

        self.positions = self.data[['Close', 'Signal', 'Position', 'Shares']]

    def simulate_trades(self):
        data = self.data
        shares = data.loc[:, 'Shares'].astype(float).squeeze()
        close = data.loc[:, 'Close'].astype(float).squeeze()
        data['Holdings'] = shares * close

        # Detect trades when position changes
        data['Trade'] = data['Position'].diff().fillna(0).abs()

        # Apply transaction cost on trades
        data['Transaction_Costs'] = data['Trade'] * self.transaction_cost * data['Holdings']

        # Cash calculation
        data['Cash'] = self.initial_capital - (data['Holdings'].diff().fillna(0) + data['Transaction_Costs'].fillna(0)).cumsum()

        # Total portfolio value
        data['Total'] = data['Cash'] + data['Holdings']

        self.portfolio = data[['Close', 'Position', 'Shares', 'Holdings', 'Cash', 'Transaction_Costs', 'Total']]

    def calculate_performance(self):
        data = self.portfolio.copy()
        data['Daily_Returns'] = data['Total'].pct_change().fillna(0)

        total_return = (data['Total'].iloc[-1] - self.initial_capital) / self.initial_capital

        sharpe_ratio = (data['Daily_Returns'].mean() / data['Daily_Returns'].std()) * np.sqrt(252) if data['Daily_Returns'].std() != 0 else 0.0

        cumulative_max = data['Total'].cummax()
        drawdown = (data['Total'] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()

        self.performance = {
            "Total Return": total_return,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown
        }

    def plot_results(self):
        data = self.portfolio.copy()
        plt.figure(figsize=(14, 7))

        plt.subplot(2, 1, 1)
        plt.plot(data['Total'], label='Portfolio Value')
        plt.plot(data['Close'], label='Price', alpha=0.5)
        plt.title('Portfolio Value vs Asset Price')
        plt.legend()
        plt.grid()

        plt.subplot(2, 1, 2)
        buy_signals = data[data['Position'].diff() > 0]
        sell_signals = data[data['Position'].diff() < 0]

        plt.plot(data['Close'], label='Price', alpha=0.7)
        plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='g', label='Buy Signal')
        plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='r', label='Sell Signal')
        plt.title('Buy/Sell Signals')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()

    def run_backtest(self):
        self.generate_positions()
        self.simulate_trades()
        self.calculate_performance()
        return self.portfolio
   
   
def compute_sma_strategy(data, short_window, long_window):
    data['SMA_short'] = data['Close'].rolling(window=short_window).mean()
    data['SMA_long'] = data['Close'].rolling(window=long_window).mean()

    data['Signal'] = 0
    data.loc[data.index[short_window:], 'Signal'] = np.where(
        data['SMA_short'][short_window:] > data['SMA_long'][short_window:], 1, -1
    )

    data['Position'] = data['Signal'].shift()
    data.dropna(inplace=True)

    # Returns
    data['Market Return'] = data['Close'].pct_change()
    data['Strategy Return'] = data['Position'] * data['Market Return']

    # Cumulative returns
    data['Cumulative Market'] = (1 + data['Market Return']).cumprod()
    data['Cumulative Strategy'] = (1 + data['Strategy Return']).cumprod()

    return data

def compute_metrics(data):
    strategy_return = data['Cumulative Strategy'].iloc[-1] - 1
    market_return = data['Cumulative Market'].iloc[-1] - 1
    if data['Strategy Return'].std() != 0:
        sharpe_ratio = np.sqrt(252) * data['Strategy Return'].mean() / data['Strategy Return'].std()
    else:
        sharpe_ratio = 0.0
    cummax = data['Cumulative Strategy'].cummax()
    drawdown = ((cummax - data['Cumulative Strategy']) / cummax).max()
    return strategy_return, market_return, sharpe_ratio, drawdown * 100

# ----------------------------------------
# Streamlit UI
# ----------------------------------------

st.set_page_config(page_title="SMA Strategy Backtester", layout="wide")

st.title("📈 SMA Crossover Strategy Backtester")
st.markdown("""
Test a **Simple Moving Average (SMA) crossover strategy** on historical stock data.  
Adjust parameters in the sidebar and run your backtest!
""")

# Sidebar Inputs
st.sidebar.header("Backtest Parameters")
ticker = st.sidebar.text_input("Stock Ticker", value="SPY")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-01-01"))
short_window = st.sidebar.number_input("Short SMA Window", min_value=2, max_value=100, value=20)
long_window = st.sidebar.number_input("Long SMA Window", min_value=5, max_value=300, value=50)

if st.sidebar.button("Run Backtest"):
    with st.spinner("Downloading data and computing strategy..."):
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                st.error("⚠️ No data found. Check your ticker or date range.")
            else:
                data = compute_sma_strategy(data, short_window, long_window)
                strategy_return, market_return, sharpe, drawdown = compute_metrics(data)

                # Backtest using Backtester class
                backtester = Backtester(data, data['Signal'])
                portfolio = backtester.run_backtest()

                # Plot cumulative returns
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(data.index, data['Cumulative Market'], label="Buy & Hold", color='blue')
                ax.plot(data.index, data['Cumulative Strategy'], label="SMA Strategy", color='green')
                ax.set_title("Cumulative Returns")
                ax.set_ylabel("Cumulative Return")
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.legend()
                st.pyplot(fig)

                # Show performance metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("📈 Buy & Hold", f"{market_return * 100:.2f}%")
                col2.metric("💡 Strategy", f"{strategy_return * 100:.2f}%")
                col3.metric("⚖️ Sharpe Ratio", f"{sharpe:.2f}")
                col4.metric("📉 Max Drawdown", f"{drawdown:.2f}%")

                # Initial capital display
                st.metric("💡 Initial Capital", f"${backtester.initial_capital:,}")

                # Optionally show raw data
                with st.expander("🔍 Show Raw Data"):
                    st.dataframe(data.tail(30))

        except Exception as e:
            st.error(f"An error occurred: {e}")