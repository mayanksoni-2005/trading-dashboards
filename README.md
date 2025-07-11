# 📈 Quantitative Trading Dashboards

A collection of interactive **Streamlit apps** developed in **Summer 2025** to simulate and analyze quantitative trading strategies and portfolio risk using real market data.

---

## 🧠 App Overview

### 1️⃣ `app.py` — **Strategy Simulator & Selector**
- Choose between:
  - **SMA Crossover**
  - **RSI Strategy**
  - **Bollinger Bands**
- Visualize technical indicators and trade signals
- Simulate simple trading logic with portfolio value tracking
- Compute basic final value (cash + holdings)

---

### 2️⃣ `sim.py` — **Monte Carlo Simulation (GBM + VaR/CVaR)**
- Simulates **stock price paths** using **Geometric Brownian Motion**
- Estimates:
  - **Value at Risk (VaR)** at 95% confidence
  - **Conditional VaR (CVaR)**
- Visualizes simulation results and loss distribution

---

### 3️⃣ `gbm.py` — **Simplified GBM Price Path Generator**
- Similar to `sim.py` but focuses more on:
  - Simulating **price evolution** using GBM
  - Visualizing potential price trajectories
- Lighter interface for pure stochastic modeling

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/mayanksoni-2005/trading-dashboards.git
cd quant-trading-dashboards
```

### 2. Set up Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Apps

| App | Purpose | Run Command |
|-----|---------|-------------|
| `app.py` | Strategy selector & trading simulator | `streamlit run app.py` |
| `sim.py` | GBM + VaR/CVaR risk simulator | `streamlit run sim.py` |
| `gbm.py` | Simplified GBM path generator | `streamlit run gbm.py` |

---

## 📦 `requirements.txt`

```txt
streamlit>=1.30.0
yfinance>=0.2.40
pandas>=2.2.2
numpy>=1.26.4
matplotlib>=3.8.4
```

---
