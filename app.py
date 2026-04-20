import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Health Diagnostic Tool",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Health Diagnostic Tool")
st.markdown("*Compare, analyse, and simulate stock portfolios. Data source: Yahoo Finance*")

# ── Sidebar ──────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
user_input = st.sidebar.text_input("Enter tickers (comma separated)", "AAPL,MSFT,NVDA,TSLA")
TICKERS = [t.strip().upper() for t in user_input.split(",") if t.strip()]
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2026-04-20"))
risk_free = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0) / 100
benchmark = st.sidebar.selectbox("Benchmark Index", ["^GSPC (S&P 500)", "^IXIC (NASDAQ)", "^DJI (Dow Jones)"])
benchmark_ticker = benchmark.split(" ")[0]
forecast_ticker = st.sidebar.selectbox("Forecast Stock", TICKERS)
st.sidebar.markdown("---")
st.sidebar.caption("ACC102 Mini Assignment | Data: Yahoo Finance")

# ── Load Data ────────────────────────────────────────────────────
@st.cache_data
def load_data(tickers, benchmark, start, end):
    all_tickers = tickers + [benchmark]
    raw = yf.download(all_tickers, start=start, end=end)
    close = raw["Close"].dropna()
    volume = raw["Volume"].dropna()
    if hasattr(close.columns, 'levels'):
        close.columns = close.columns.get_level_values(0)
        volume.columns = volume.columns.get_level_values(0)
    return close, volume

with st.spinner("Loading data from Yahoo Finance..."):
    try:
        close_df, volume_df = load_data(TICKERS, benchmark_ticker, str(start_date), str(end_date))
        st.success(f"✅ Loaded {len(close_df)} trading days | {close_df.index[0].date()} to {close_df.index[-1].date()}")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# ── Compute Stats ─────────────────────────────────────────────────
def compute_stats(ticker):
    prices = close_df[ticker].dropna()
    daily_ret = prices.pct_change().dropna()
    total_ret = (prices.iloc[-1] / prices.iloc[0]) - 1
    ann_ret = (1 + total_ret) ** (252 / len(prices)) - 1
    ann_vol = daily_ret.std() * np.sqrt(252)
    rolling_max = prices.cummax()
    max_dd = ((prices - rolling_max) / rolling_max).min()
    sharpe = (ann_ret - risk_free) / ann_vol
    return ann_ret, ann_vol, max_dd, sharpe, prices, daily_ret

# ── Scoring ───────────────────────────────────────────────────────
def score_stock(ann_ret, ann_vol, max_dd, sharpe):
    score = 0
    score += min(max(ann_ret * 100, 0), 30)
    score += min(max(sharpe * 15, 0), 30)
    score += max(20 - ann_vol * 40, 0)
    score += max(20 + max_dd * 30, 0)
    return round(min(score, 100))

def get_label(score):
    if score >= 65:
        return "🟢 Aggressive / Growth"
    elif score >= 40:
        return "🟡 Balanced / Moderate"
    else:
        return "🔴 Conservative / Defensive"

def get_conclusion(ticker, ann_ret, ann_vol, max_dd, sharpe, score):
    ret_desc = "high return" if ann_ret > 0.15 else "moderate return" if ann_ret > 0.05 else "low return"
    risk_desc = "high risk" if ann_vol > 0.40 else "moderate risk" if ann_vol > 0.25 else "low risk"
    sharpe_desc = "strong risk-adjusted performance" if sharpe > 0.8 else "average risk-adjusted performance" if sharpe > 0.3 else "weak risk-adjusted performance"
    return (
        f"**{ticker}** is a **{risk_desc}, {ret_desc}** stock with {sharpe_desc}. "
        f"Maximum drawdown: **{max_dd:.1%}**. Overall score: **{score}/100**. "
        f"Profile: {get_label(score)}."
    )

# ── Tabs ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Statistics & Scores",
    "📈 Price & Volume",
    "⚠️ Risk Analysis",
    "🔥 Correlation",
    "💼 Portfolio Simulator",
    "🔮 Forecast"
])

# ════════════════════════════════════════════════════════════════
# TAB 1: Key Statistics + Scoring
# ════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Key Statistics, Scores & Investment Profile")
    results = []
    conclusions = []
    for ticker in TICKERS:
        if ticker not in close_df.columns:
            continue
        ann_ret, ann_vol, max_dd, sharpe, prices, daily_ret = compute_stats(ticker)
        score = score_stock(ann_ret, ann_vol, max_dd, sharpe)
        results.append({
            "Ticker": ticker,
            "Ann. Return": f"{ann_ret:.2%}",
            "Ann. Volatility": f"{ann_vol:.2%}",
            "Max Drawdown": f"{max_dd:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Score (0-100)": score,
            "Profile": get_label(score)
        })
        conclusions.append(get_conclusion(ticker, ann_ret, ann_vol, max_dd, sharpe, score))

    st.dataframe(pd.DataFrame(results).set_index("Ticker"), use_container_width=True)
    st.markdown("---")
    st.subheader("🤖 Auto-Generated Conclusions")
    for c in conclusions:
        st.markdown(f"- {c}")
    st.caption("Score = weighted combination of annualized return, Sharpe ratio, volatility, and max drawdown.")

# ════════════════════════════════════════════════════════════════
# TAB 2: Price & Volume + Benchmark
# ════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Normalized Price vs Benchmark")
    available = [t for t in TICKERS if t in close_df.columns]
    bench_available = benchmark_ticker in close_df.columns

    normalized = close_df[available] / close_df[available].iloc[0] * 100
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['steelblue', 'darkorange', 'green', 'red', 'purple', 'brown']
    for i, ticker in enumerate(available):
        ax.plot(normalized.index, normalized[ticker], label=ticker,
                linewidth=1.5, color=colors[i % len(colors)])
    if bench_available:
        bench_norm = close_df[benchmark_ticker] / close_df[benchmark_ticker].iloc[0] * 100
        ax.plot(bench_norm.index, bench_norm, label=f"Benchmark ({benchmark_ticker})",
                linewidth=2, color='black', linestyle='--')
    ax.axhline(y=100, color='gray', linestyle=':', linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Price (Base=100)")
    ax.set_title("Stock Performance vs Benchmark")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    if bench_available:
        st.markdown("#### 📐 Alpha vs Benchmark")
        bench_ret = (close_df[benchmark_ticker].iloc[-1] / close_df[benchmark_ticker].iloc[0]) - 1
        bench_ann = (1 + bench_ret) ** (252 / len(close_df)) - 1
        alpha_data = []
        for ticker in available:
            ann_ret, _, _, _, _, _ = compute_stats(ticker)
            alpha = ann_ret - bench_ann
            alpha_data.append({
                "Ticker": ticker,
                "Stock Return": f"{ann_ret:.2%}",
                "Benchmark Return": f"{bench_ann:.2%}",
                "Alpha": f"{alpha:.2%}",
                "Beat Market?": "✅ Yes" if alpha > 0 else "❌ No"
            })
        st.dataframe(pd.DataFrame(alpha_data).set_index("Ticker"), use_container_width=True)
        st.caption(f"Alpha = Stock annualized return minus {benchmark_ticker} annualized return.")

    st.markdown("---")
    st.subheader("📦 Price & Volume Relationship")
    vol_select = st.selectbox("Select ticker", available, key="vol_sel")
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.plot(close_df[vol_select].index, close_df[vol_select], color='steelblue', linewidth=1.2)
    ax1.set_ylabel("Price (USD)")
    ax1.set_title(f"{vol_select} — Price & Volume")
    if vol_select in volume_df.columns:
        ax2.bar(volume_df[vol_select].index, volume_df[vol_select],
                color='steelblue', alpha=0.5, width=1)
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Date")
    plt.tight_layout()
    st.pyplot(fig2)
    st.caption("Volume spikes often coincide with major price movements — a key signal for traders.")

# ════════════════════════════════════════════════════════════════
# TAB 3: Risk Analysis
# ════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Risk Analysis")
    available = [t for t in TICKERS if t in close_df.columns]
    risk_sel = st.selectbox("Select ticker", available, key="risk_sel")
    ann_ret, ann_vol, max_dd, sharpe, prices, daily_ret = compute_stats(risk_sel)

    st.markdown("#### 📉 Rolling 30-Day Volatility")
    rolling_vol = daily_ret.rolling(30).std() * np.sqrt(252)
    fig3, ax = plt.subplots(figsize=(12, 3))
    ax.plot(rolling_vol.index, rolling_vol, color='darkorange', linewidth=1.2)
    ax.set_ylabel("Annualized Volatility")
    ax.set_title(f"{risk_sel} — Rolling 30-Day Volatility")
    ax.set_xlabel("Date")
    plt.tight_layout()
    st.pyplot(fig3)
    st.caption("Spikes indicate periods of market stress or major news events.")

    st.markdown("#### 📉 Maximum Drawdown Curve")
    drawdown_curve = (prices - prices.cummax()) / prices.cummax()
    fig4, ax = plt.subplots(figsize=(12, 3))
    ax.fill_between(drawdown_curve.index, drawdown_curve, 0, color='red', alpha=0.4)
    ax.plot(drawdown_curve.index, drawdown_curve, color='red', linewidth=0.8)
    ax.set_ylabel("Drawdown")
    ax.set_title(f"{risk_sel} — Drawdown from Peak (Max: {max_dd:.2%})")
    ax.set_xlabel("Date")
    plt.tight_layout()
    st.pyplot(fig4)

    st.markdown("#### 📊 Daily Return Distribution")
    fig5, ax = plt.subplots(figsize=(8, 4))
    ax.hist(daily_ret, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(daily_ret.mean(), color='red', linestyle='--', label=f'Mean: {daily_ret.mean():.4f}')
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{risk_sel} — Daily Return Distribution")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig5)

    col1, col2 = st.columns(2)
    col1.metric("Skewness", f"{daily_ret.skew():.3f}", help="Negative = more extreme losses than gains")
    col2.metric("Kurtosis", f"{daily_ret.kurtosis():.3f}", help="High = fat tails = more extreme events")
    st.caption("Normal distribution has skewness ≈ 0, kurtosis ≈ 0. Fat tails indicate higher crash risk.")

# ════════════════════════════════════════════════════════════════
# TAB 4: Correlation
# ════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Return Correlation Matrix")
    available = [t for t in TICKERS if t in close_df.columns]
    corr = close_df[available].pct_change().dropna().corr()
    fig6, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    tl = corr.columns.tolist()
    ax.set_xticks(range(len(tl)))
    ax.set_yticks(range(len(tl)))
    ax.set_xticklabels(tl)
    ax.set_yticklabels(tl)
    for i in range(len(tl)):
        for j in range(len(tl)):
            val = corr.iloc[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='black' if abs(val) < 0.7 else 'white')
    ax.set_title("Return Correlation Matrix")
    plt.tight_layout()
    st.pyplot(fig6)
    st.caption("Values closer to 1.0 = stocks move together. Lower correlation = better diversification.")

# ════════════════════════════════════════════════════════════════
# TAB 5: Portfolio Simulator
# ════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("💼 Portfolio Simulator")
    st.markdown("Allocate weights to each stock and see your portfolio's combined performance vs the benchmark.")

    available = [t for t in TICKERS if t in close_df.columns]
    n = len(available)

    st.markdown("#### Set Portfolio Weights")
    cols = st.columns(n)
    weights = []
    for i, ticker in enumerate(available):
        w = cols[i].slider(f"{ticker} (%)", 0, 100, 100 // n, key=f"w_{ticker}")
        weights.append(w)

    total_weight = sum(weights)
    st.markdown(f"**Total allocated: {total_weight}%**")

    if total_weight != 100:
        st.warning(f"⚠️ Weights sum to {total_weight}%. Please adjust sliders to exactly 100%.")
    else:
        weights_norm = [w / 100 for w in weights]
        returns_df = close_df[available].pct_change().dropna()
        portfolio_ret = sum(returns_df[t] * w for t, w in zip(available, weights_norm))
        portfolio_value = (1 + portfolio_ret).cumprod() * 100

        bench_available = benchmark_ticker in close_df.columns
        if bench_available:
            bench_ret_series = close_df[benchmark_ticker].pct_change().dropna()
            bench_value = (1 + bench_ret_series).cumprod() * 100

        total_ret = portfolio_value.iloc[-1] / 100 - 1
        ann_ret = (1 + total_ret) ** (252 / len(portfolio_ret)) - 1
        ann_vol = portfolio_ret.std() * np.sqrt(252)
        sharpe_p = (ann_ret - risk_free) / ann_vol
        max_dd_p = ((portfolio_value - portfolio_value.cummax()) / portfolio_value.cummax()).min()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ann. Return", f"{ann_ret:.2%}")
        c2.metric("Ann. Volatility", f"{ann_vol:.2%}")
        c3.metric("Max Drawdown", f"{max_dd_p:.2%}")
        c4.metric("Sharpe Ratio", f"{sharpe_p:.2f}")

        fig7, ax = plt.subplots(figsize=(12, 5))
        ax.plot(portfolio_value.index, portfolio_value,
                label="Your Portfolio", linewidth=2, color='steelblue')
        if bench_available:
            ax.plot(bench_value.index, bench_value,
                    label=f"Benchmark ({benchmark_ticker})",
                    linewidth=1.5, linestyle='--', color='black')
        ax.axhline(y=100, color='gray', linestyle=':', linewidth=0.8)
        ax.set_title("Portfolio Growth vs Benchmark (Base = 100)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig7)

        col_pie, col_info = st.columns([1, 1])
        with col_pie:
            fig8, ax2 = plt.subplots(figsize=(5, 5))
            ax2.pie(weights_norm, labels=available, autopct='%1.1f%%',
                    colors=['steelblue', 'darkorange', 'green', 'red', 'purple', 'brown'][:n])
            ax2.set_title("Portfolio Allocation")
            st.pyplot(fig8)

        with col_info:
            if bench_available:
                bench_total = bench_value.iloc[-1] / 100 - 1
                bench_ann = (1 + bench_total) ** (252 / len(bench_ret_series)) - 1
                alpha = ann_ret - bench_ann
                st.markdown("#### vs Benchmark")
                st.metric("Portfolio Return", f"{ann_ret:.2%}")
                st.metric("Benchmark Return", f"{bench_ann:.2%}")
                st.metric("Alpha", f"{alpha:.2%}",
                          delta=f"{'Outperforming ✅' if alpha > 0 else 'Underperforming ❌'}")

# ════════════════════════════════════════════════════════════════
# TAB 6: Forecast
# ════════════════════════════════════════════════════════════════
with tab6:
    st.subheader(f"🔮 Linear Regression Forecast: {forecast_ticker}")
    if forecast_ticker not in close_df.columns:
        st.error(f"{forecast_ticker} not available in loaded data.")
    else:
        ann_ret, ann_vol, max_dd, sharpe, prices, daily_ret = compute_stats(forecast_ticker)
        X = np.arange(len(prices)).reshape(-1, 1)
        y = prices.values
        model = LinearRegression()
        model.fit(X, y)
        future_X = np.arange(len(prices), len(prices) + 30).reshape(-1, 1)
        future_pred = model.predict(future_X)
        future_dates = pd.bdate_range(start=prices.index[-1], periods=31)[1:]

        fig9, ax = plt.subplots(figsize=(12, 5))
        ax.plot(prices.index[-120:], prices[-120:], label='Actual Price',
                linewidth=1.5, color='steelblue')
        ax.plot(future_dates, future_pred, label='Forecast (30 days)',
                linestyle='--', color='orange', linewidth=2)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.set_title(f"{forecast_ticker} — 30-Day Price Forecast (Linear Regression)")
        ax.legend()
        ax.text(0.01, 0.05, '* For educational purposes only. Not financial advice.',
                transform=ax.transAxes, fontsize=8, color='gray')
        plt.tight_layout()
        st.pyplot(fig9)

        col1, col2, col3 = st.columns(3)
        change = (future_pred[-1] / prices.iloc[-1]) - 1
        col1.metric("Current Price", f"${prices.iloc[-1]:.2f}")
        col2.metric("Predicted (30d)", f"${future_pred[-1]:.2f}")
        col3.metric("Expected Change", f"{change:.2%}", delta=f"{change:.2%}")

        st.warning("⚠️ Linear regression captures long-term trend only. It does not account for earnings, macro events, or market sentiment. For educational purposes only.")
