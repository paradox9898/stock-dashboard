import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser

st.set_page_config(page_title="AI Stock Dashboard", layout="wide")

# -----------------------------
# INDICATORS
# -----------------------------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def vwap(df):
    volume_sum = df["Volume"].cumsum()
    price_volume_sum = (df["Close"] * df["Volume"]).cumsum()
    return price_volume_sum / volume_sum

# -----------------------------
# DATA FETCH
# -----------------------------
@st.cache_data(ttl=900)
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="6mo", interval="1d")

        if df.empty:
            return None, None, "No price data returned for this ticker."

        df = df.dropna().copy()
        df["EMA20"] = ema(df["Close"], 20)
        df["EMA50"] = ema(df["Close"], 50)
        df["EMA200"] = ema(df["Close"], 200)
        df["RSI"] = rsi(df["Close"])
        df["VWAP"] = vwap(df)

        try:
            info = stock.info
        except Exception:
            info = {}

        return df, info, None

    except Exception as e:
        return None, None, str(e)

# -----------------------------
# NEWS
# -----------------------------
@st.cache_data(ttl=1800)
def get_news(ticker):
    try:
        url = f"https://news.google.com/rss/search?q={ticker}+stock"
        feed = feedparser.parse(url)
        news = []

        for entry in feed.entries[:10]:
            news.append({
                "title": entry.title,
                "link": entry.link
            })

        return news
    except Exception:
        return []

# -----------------------------
# SIMPLE SENTIMENT PLACEHOLDER
# -----------------------------
def get_sentiment_stub():
    return {
        "Bullish Score": "Free version placeholder",
        "Bearish Score": "Free version placeholder",
        "Status": "Reddit/StockTwits sentiment can be added later"
    }

# -----------------------------
# ANALYSIS ENGINE
# -----------------------------
def analyze(df, info, ticker):
    latest = df.iloc[-1]

    price = float(latest["Close"])
    ema20_val = float(latest["EMA20"])
    ema50_val = float(latest["EMA50"])
    ema200_val = float(latest["EMA200"])
    rsi_val = float(latest["RSI"])

    revenue = info.get("totalRevenue", None)
    debt = info.get("totalDebt", None)
    cash = info.get("totalCash", None)
    long_name = info.get("longName", ticker.upper())

    # Fundamentals
    fundamentals_summary = "Limited free data available."
    fundamentals_score = 3

    if revenue and cash is not None and debt is not None:
        if cash > debt:
            fundamentals_summary = (
                f"{long_name} shows decent balance sheet strength with cash exceeding debt. "
                f"Revenue data is available, which supports a more stable fundamental profile."
            )
            fundamentals_score = 4
        else:
            fundamentals_summary = (
                f"{long_name} has revenue data available, but debt appears higher relative to cash, "
                f"so balance-sheet risk is higher."
            )
            fundamentals_score = 2

    # Technicals
    technicals_score = 3
    technicals_summary = (
        f"Price is {price:.2f}. EMA20 is {ema20_val:.2f}, EMA50 is {ema50_val:.2f}, "
        f"EMA200 is {ema200_val:.2f}, and RSI is {rsi_val:.2f}."
    )

    if price > ema50_val and price > ema200_val:
        technicals_score = 4
        technicals_summary += " Trend structure is bullish because price is above key moving averages."

    if rsi_val > 70:
        technicals_summary += " RSI suggests the stock may be overbought."
    elif rsi_val < 30:
        technicals_summary += " RSI suggests the stock may be oversold."

    # Risk Management
    daily_returns = df["Close"].pct_change().dropna()
    volatility = float(daily_returns.std()) if not daily_returns.empty else 0

    if volatility < 0.02:
        risk_score = 4
        risk_summary = (
            "Volatility is relatively moderate for a stock, which supports more controlled position sizing."
        )
    elif volatility < 0.04:
        risk_score = 3
        risk_summary = (
            "Volatility is moderate. Position sizing should stay disciplined, especially around news events."
        )
    else:
        risk_score = 2
        risk_summary = (
            "Volatility is high. This stock may require smaller position sizing and wider stop placement."
        )

    # Trading Plan
    if price > ema200_val:
        trading_plan_score = 4
        trading_plan_summary = (
            "Primary thesis is trend continuation. A 6-12 month view favors holding while the long-term trend remains intact."
        )
    else:
        trading_plan_score = 2
        trading_plan_summary = (
            "Primary thesis is weaker because the stock is below its long-term trend. A 6-12 month outlook needs caution."
        )

    # Entry / Exit
    recent_low = float(df["Low"].tail(20).min())
    recent_high = float(df["High"].tail(20).max())

    entry_zone_low = min(price, ema50_val)
    entry_zone_high = max(price, ema50_val)
    target_price = max(price * 1.15, recent_high * 1.05)
    stop_loss = min(recent_low, price * 0.92)

    entry_exit_score = 3
    entry_exit_summary = (
        f"Possible entry zone: {entry_zone_low:.2f} to {entry_zone_high:.2f}. "
        f"12-month target price: {target_price:.2f}. "
        f"Protective stop-loss: {stop_loss:.2f}."
    )

    # Strong Mindset
    if volatility >= 0.04:
        mindset_score = 3
        mindset_summary = (
            "Main emotional pitfall is reacting to volatility. Rules: avoid revenge trading, respect stop-losses, and keep size small."
        )
    else:
        mindset_score = 4
        mindset_summary = (
            "Main emotional pitfall is overconfidence during steady trends. Rules: stick to the plan, avoid chasing, and review risk before adding."
        )

    scores = {
        "Fundamentals": fundamentals_score,
        "Technicals": technicals_score,
        "Risk Management": risk_score,
        "Trading Plan": trading_plan_score,
        "Entry/Exit Strategy": entry_exit_score,
        "Strong Mindset": mindset_score,
    }

    overall_score = round(sum(scores.values()) / len(scores), 2)

    return {
        "fundamentals_summary": fundamentals_summary,
        "technicals_summary": technicals_summary,
        "risk_summary": risk_summary,
        "trading_plan_summary": trading_plan_summary,
        "entry_exit_summary": entry_exit_summary,
        "mindset_summary": mindset_summary,
        "scores": scores,
        "overall_score": overall_score,
    }

# -----------------------------
# UI
# -----------------------------
st.title("AI Stock Intelligence Dashboard")

st.write("Enter a ticker to load price data, news, and a free 6-pillar analysis.")

ticker = st.text_input("Enter Ticker (e.g., NVDA, AAPL)", "").strip().upper()

if not ticker:
    st.info("Enter a ticker above to begin.")
    st.stop()

df, info, error = get_stock_data(ticker)

if error:
    st.error(f"Data temporarily unavailable: {error}")
    st.info("This usually happens because Yahoo Finance free data is rate-limited. Wait a minute and try again.")
    st.stop()

analysis = analyze(df, info or {}, ticker)
news = get_news(ticker)
sentiment = get_sentiment_stub()

# -----------------------------
# TOP METRICS
# -----------------------------
latest_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else latest_close
change = latest_close - prev_close
pct_change = (change / prev_close * 100) if prev_close != 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Ticker", ticker)
col2.metric("Last Price", f"{latest_close:.2f}", f"{change:.2f} ({pct_change:.2f}%)")
col3.metric("Overall Score", f"{analysis['overall_score']} / 5")

# -----------------------------
# CHARTS
# -----------------------------
st.subheader("Price and Moving Averages")
st.line_chart(df[["Close", "EMA20", "EMA50", "EMA200"]])

st.subheader("RSI")
st.line_chart(df[["RSI"]])

# -----------------------------
# NEWS + SENTIMENT
# -----------------------------
left, right = st.columns(2)

with left:
    st.subheader("Latest News")
    if news:
        for item in news[:8]:
            st.markdown(f"- [{item['title']}]({item['link']})")
    else:
        st.write("No news available right now.")

with right:
    st.subheader("Sentiment")
    for k, v in sentiment.items():
        st.write(f"**{k}:** {v}")

# -----------------------------
# 6-PILLAR ANALYSIS
# -----------------------------
st.subheader("Comprehensive 6-Pillar Trading Analysis")

st.markdown("**1. Fundamentals**")
st.write(analysis["fundamentals_summary"])

st.markdown("**2. Technicals**")
st.write(analysis["technicals_summary"])

st.markdown("**3. Risk Management**")
st.write(analysis["risk_summary"])

st.markdown("**4. Trading Plan**")
st.write(analysis["trading_plan_summary"])

st.markdown("**5. Entry/Exit Strategy**")
st.write(analysis["entry_exit_summary"])

st.markdown("**6. Strong Mindset**")
st.write(analysis["mindset_summary"])

# -----------------------------
# RATING TABLE
# -----------------------------
st.subheader("Consolidated Rating Table")

rating_df = pd.DataFrame(
    {
        "Pillar": list(analysis["scores"].keys()),
        "Score (1-5)": list(analysis["scores"].values()),
    }
)

st.dataframe(rating_df, use_container_width=True)
st.success(f"Overall Score: {analysis['overall_score']} / 5")
