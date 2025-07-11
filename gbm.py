import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("Monte Carlo Simulation for Stock Prices with GBM")

# User inputs
stock = st.text_input("Enter Stock Ticker (e.g. AAPL)", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-01-01"))

num_simulations = st.number_input("Number of Simulations", min_value=100, max_value=10000, value=500, step=100)
time_horizon = st.number_input("Time Horizon (days)", min_value=10, max_value=2520, value=252)
risk_free_rate = st.number_input("Risk-free rate (annual, in decimal)", min_value=0.0, max_value=0.1, value=0.01, step=0.001)

if start_date >= end_date:
    st.error("Start Date must be before End Date.")
else:
    # Download historical stock data
    data = yf.download(stock, start=start_date, end=end_date)

    if data.empty:
        st.error("No data found for the ticker and date range. Please check inputs.")
    else:
        # Calculate daily log returns
        data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
        data.dropna(inplace=True)

        # Estimate parameters for GBM
        mu = data['log_return'].mean() * 252  # annualized mean return
        sigma = data['log_return'].std() * np.sqrt(252)  # annualized volatility

        st.write(f"Estimated annual return (mu): {mu:.4f}")
        st.write(f"Estimated annual volatility (sigma): {sigma:.4f}")

        # Initialize simulation array: shape (time_horizon, num_simulations)
        simulations = np.zeros((time_horizon, num_simulations))

        # Set initial price for all simulations to last observed close price
        last_price = data['Close'].iloc[-1]
        simulations[0] = last_price

        dt = 1 / 252  # daily timestep

        # Run simulation
        for t in range(1, time_horizon):
            z = np.random.standard_normal(num_simulations)  # random shocks
            simulations[t] = simulations[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)

        # Plot some simulation paths
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(min(num_simulations, 20)):  # plot up to 20 paths
            ax.plot(simulations[:, i], lw=0.8, alpha=0.7)
        ax.set_title(f"Monte Carlo Simulated Stock Price Paths for {stock}")
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        st.pyplot(fig)

        # Calculate Value at Risk (VaR) at 95% confidence for final day
        ending_prices = simulations[-1]
        var_95 = np.percentile(ending_prices, 5)
        cvar_95 = ending_prices[ending_prices <= var_95].mean()
        
        last_price_scalar = last_price.iloc[0]  # or .values[0]

        st.write(f"Estimated 95% VaR: {last_price_scalar - var_95:.2f} (Loss)")
        st.write(f"Estimated 95% CVaR: {last_price_scalar - cvar_95:.2f} (Average loss beyond VaR)")
