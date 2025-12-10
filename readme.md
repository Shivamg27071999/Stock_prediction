# 📈 Stock Price Prediction Dashboard (Machine Learning + Streamlit)

This is an interactive machine-learning web application for **stock price prediction**, built using **Streamlit**, **scikit-learn**, **Plotly**, and **yfinance**.  
The goal of this project is to forecast the next 5 trading days, visualize model performance, and experiment with technical indicators and ML models in a clean dashboard UI.

---

## 🚀 Features

### ✔️ Live Stock Data Fetching
- Pulls live historical OHLCV data from Yahoo Finance.
- Works with NSE, BSE, NASDAQ, NYSE tickers (e.g., `RELIANCE.NS`, `TCS.NS`, `AAPL`, etc.)

### ✔️ Technical Indicators Supported
- SMA (5, 20)
- EMA (12, 26)
- MACD + Signal
- RSI-14
- Bollinger Bands
- OBV (On-Balance Volume)

### ✔️ Lag-Based Feature Engineering
- Adjustable lag window (3–20 days)
- Auto-aligned feature matrix and target series

### ✔️ Machine Learning Models Included
- **Ridge Regression**
- **Random Forest Regressor**
- Optional **AutoML hyperparameter tuning**  
- Optional **Model Ensemble** mode

### ✔️ Backtesting (Expanding Window)
The app shows:
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)
- Predicted vs Actual plot
- Residual distribution chart

### ✔️ Next-5 Day Prediction
Produces:
- Predicted Close prices  
- Or next-day return % if selected  
- Based on iterative forecasting using trained model

### ✔️ Modern Custom UI
Custom CSS gives the dashboard:
- Dark gradient background  
- Light cards with readable text  
- High-contrast Plotly graphs  
- Input elements with clean minimal styling  
- Footer: “Built by Shivam” + GitHub & LinkedIn links  

---

## 🧠 How It Works

1. User selects a stock ticker and time period.  
2. Historical data is fetched using `yfinance`.  
3. Technical indicators + lag features are generated.  
4. Model is trained on expanding-window fashion.  
5. Backtest predictions help evaluate performance.  
6. The final trained model forecasts the next 5 trading days.  
7. Charts and metrics update automatically based on user inputs.

---

## 🛠️ Technologies Used

### Backend / ML
- **Python**
- **scikit-learn**
- **pandas**
- **numpy**
- **yfinance**

### Frontend / UI
- **Streamlit**
- **Plotly**
- **Custom CSS**

---

## 📦 Installation & Setup

### Step 1 — Clone the repository
```bash
git clone https://github.com/Shivamg27071999/Stock_prediction.git
cd Stock_prediction
