# Stock Health Diagnostic Tool

An interactive stock analysis application built with Python and Streamlit. It allows retail investors to compare stock performance, analyse risk, simulate custom portfolios, and generate price forecasts without requiring any financial expertise.

**Analytical Problem:** Retail investors often struggle to assess stock performance beyond simple price movements. This tool computes standard financial metrics including annualised return, volatility, maximum drawdown, and Sharpe ratio, and presents them in an accessible, interactive format.

**Target User:** Individual retail investors with no professional finance background who want a data-informed view of stock performance.


## Features

| Tab | Description |
|-----|-------------|
| Statistics and Scores | Key financial metrics for each stock, a composite score from 0 to 100, investment profile classification, and auto-generated plain-English conclusions |
| Price and Volume | Normalised price comparison against a benchmark index, alpha table showing which stocks beat the market, and a price-volume relationship chart |
| Risk Analysis | Rolling 30-day volatility, maximum drawdown curve, and daily return distribution with skewness and kurtosis |
| Correlation | Return correlation heatmap showing diversification relationships between selected stocks |
| Portfolio Simulator | Custom weight allocation via sliders with real-time portfolio statistics and benchmark comparison |
| Forecast | 30-day linear regression price trend forecast with educational disclaimer |


## Data Source

Data are sourced from Yahoo Finance via the yfinance Python library.

| Item | Detail |
|------|--------|
| Tickers | AAPL, MSFT, NVDA, TSLA, and ^GSPC (S&P 500 benchmark) |
| Date range | January 2022 to April 2026 |
| Access date | April 2026 |
| Data type | Daily closing prices and trading volumes |
| Storage | Pre-exported as close_prices.csv and volume_data.csv for stable deployment |

This date range captures a full market cycle including the 2022 rate-hike-driven bear market, the 2023 recovery, the 2024 AI-driven semiconductor rally, and the corrections of 2025 and early 2026.


## Installation and Setup

**Requirements:** Python 3.8 or above

**Step 1: Clone the repository**

```
git clone https://github.com/YOUR_USERNAME/stock-health-tool.git
cd stock-health-tool
```

**Step 2: Install dependencies**

```
pip install -r requirements.txt
```

**Step 3: Run the application**

```
streamlit run app.py
```

The app will open automatically in your browser at http://localhost:8501.

**File structure:**

```
stock-health-tool/
├── app.py                  Main Streamlit application
├── requirements.txt        Python dependencies
├── close_prices.csv        Pre-downloaded stock closing prices
├── volume_data.csv         Pre-downloaded trading volume data
├── notebook.ipynb          Jupyter Notebook with full analytical workflow
└── README.md               Project documentation
```


## How To Use It

1. Enter stock tickers in the sidebar separated by commas, for example AAPL, MSFT, NVDA, TSLA
2. Select your preferred date range using the date pickers
3. Adjust the risk-free rate slider if needed (default is 5%)
4. Choose a benchmark index to compare against: S&P 500, NASDAQ, or Dow Jones
5. Select which stock to forecast in the final tab
6. Navigate through the six tabs to explore different aspects of the analysis
7. In the Portfolio Simulator tab, drag the sliders to allocate weights and observe how your portfolio performs against the benchmark in real time


## Key Macroeconomic Events in the Analysis Period

| Period | Event | Market Impact |
|--------|-------|---------------|
| 2022 | Federal Reserve interest rate tightening cycle | Broad equity selloff, all four stocks declined significantly |
| Mid-2023 | AI demand surge driven by large language model adoption | NVDA began its exceptional rally |
| 2024 | Continued AI infrastructure investment | NVDA reached peak normalised returns above 600 |
| 2025 | Macro uncertainty and rate expectations | Renewed volatility spikes across all stocks |
| Early 2026 | Market correction | TSLA and NVDA experienced further drawdowns |


## Limitations

The dataset covers only four technology stocks, which limits generalisability across broader market segments. The linear regression forecast is a deliberate simplification that ignores mean reversion, volatility clustering, and macroeconomic shocks. A more credible forecasting approach would use ARIMA or machine learning models. The Sharpe ratio applies a static risk-free rate of 5% throughout the period, which introduces bias given that the actual rate varied substantially between 2022 and 2026. Data are pre-loaded from CSV files rather than streamed live, meaning the dataset does not update automatically beyond April 2026.


## Disclaimer

This tool is designed for educational purposes only. All scores, forecasts, and analysis are based on historical data and simplified financial models. Nothing presented here constitutes financial advice. Always consult a qualified financial professional before making investment decisions.

Built with Python and Streamlit | ACC102 Mini Assignment, XJTLU 2026
