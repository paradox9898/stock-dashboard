import math
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    price_volume_sum = ((df["High"] + df["Low"] + df["Close"]) / 3 * df["Volume"]).cumsum()
    return price_volume_sum / volume_sum

# -----------------------------
# DATA FETCH
# -----------------------------
@st.cache_data(ttl=900)
def get_stock_data(ticker, period="6mo", interval="1d"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval, auto_adjust=False)

        if df.empty:
            return None, None, "No price data returned for this ticker."

        df = df.dropna().copy()
        df["EMA9"] = ema(df["Close"], 9)
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

        for entry in feed.entries[:12]:
            news.append({
                "title": entry.title,
                "link": entry.link,
                "published": getattr(entry, "published", "")
            })

        return news
    except Exception:
        return []

# -----------------------------
# FREE HEADLINE SENTIMENT
# -----------------------------
def calculate_sentiment(news_items):
    bullish_words = {
        "beat", "beats", "surge", "up", "bullish", "growth", "strong", "upgrade",
        "record", "profit", "profits", "gain", "gains", "soar", "soars", "buy",
        "outperform", "momentum", "expands", "expansion", "positive", "rebound"
    }

    bearish_words = {
        "miss", "misses", "drop", "down", "bearish", "downgrade", "cut", "cuts",
        "warning", "weak", "fall", "falls", "lawsuit", "risk", "loss", "losses",
        "decline", "declines", "probe", "investigation", "negative", "concern"
    }

    score = 0
    total_hits = 0

    for item in news_items:
        title = item["title"].lower()
        bull_hits = sum(1 for word in bullish_words if word in title)
        bear_hits = sum(1 for word in bearish_words if word in title)
        score += bull_hits
        score -= bear_hits
        total_hits += bull_hits + bear_hits

    if total_hits == 0:
        normalized = 50
    else:
        normalized = 50 + max(-40, min(40, score * 8))

    normalized = int(max(0, min(100, normalized)))

    if normalized >= 65:
        label = "Bullish"
    elif normalized <= 35:
        label = "Bearish"
    else:
        label = "Neutral"

    return normalized, label

# -----------------------------
# SUPPORT / RESISTANCE
# -----------------------------
def get_support_resistance(df):
    recent_support = float(df["Low"].tail(20).min())
    recent_resistance = float(df["High"].tail(20).max())
    return recent_support, recent_resistance

# -----------------------------
# AI-STYLE COMMENTS
# -----------------------------
def generate_ai_commentary(ticker, price, rsi_value, overall_score, sentiment_label, sentiment_score):
    comments = []

    if overall_score >= 4:
        comments.append(f"{ticker} currently scores strongly across multiple pillars, suggesting a healthier overall setup than average.")
    elif overall_score >= 3:
        comments.append(f"{ticker} shows a mixed but tradable setup, with strengths in some areas and caution flags in others.")
    else:
        comments.append(f"{ticker} currently looks weaker on the combined model, so risk control matters more than aggressive conviction.")

    if rsi_value > 70:
        comments.append("Momentum is strong, but RSI is elevated, so chasing price without a pullback may increase risk.")
    elif rsi_value < 30:
        comments.append("RSI is in an oversold region, which can support a bounce thesis, but trend confirmation is still important.")
    else:
        comments.append("RSI is in a more balanced range, which gives flexibility for both continuation and pullback scenarios.")

    comments.append(
        f"Headline sentiment is {sentiment_label.lower()} at {sentiment_score}/100, which should be used as context, not as a stand-alone trading signal."
    )

    comments.append(
        "Best practice: wait for price confirmation near support, keep position size controlled, and avoid changing the plan emotionally after entry."
    )

    return comments

# -----------------------------
# ANALYSIS ENGINE
# -----------------------------
def analyze(df, info, ticker, sentiment_score):
    latest = df.iloc[-1]

    price = float(latest["Close"])
    ema9_val = float(latest["EMA9"])
    ema20_val = float(latest["EMA20"])
    ema50_val = float(latest["EMA50"])
    ema200_val = float(latest["EMA200"])
    rsi_val = float(latest["RSI"])

    revenue = info.get("totalRevenue", None)
    debt = info.get("totalDebt", None)
    cash = info.get("totalCash", None)
    long_name = info.get("longName", ticker.upper())

    support, resistance = get_support_resistance(df)

    # Fundamentals
    fundamentals_score = 3
    if revenue and cash is not None and debt is not None:
        if cash > debt:
            fundamentals_score = 4
            fundamentals_summary = (
                f"{long_name} appears financially steadier, with cash exceeding debt. "
                f"That supports balance-sheet durability and lowers funding stress."
            )
        else:
            fundamentals_score = 2
            fundamentals_summary = (
                f"{long_name} has accessible revenue data, but debt is heavier relative to cash, "
                f"which raises balance-sheet risk."
            )
    else:
        fundamentals_summary = (
            f"Only limited free fundamentals are available for {long_name}. "
            f"The balance-sheet picture is not complete enough for a high-conviction fundamental score."
        )

    # Technicals
    technicals_score = 3
    technicals_summary = (
        f"Price is {price:.2f}. EMA9 is {ema9_val:.2f}, EMA20 is {ema20_val:.2f}, "
        f"EMA50 is {ema50_val:.2f}, EMA200 is {ema200_val:.2f}, RSI is {rsi_val:.2f}. "
        f"Support is near {support:.2f} and resistance is near {resistance:.2f}. "
    )

    if price > ema20_val > ema50_val and price > ema200_val:
        technicals_score = 4
        technicals_summary += "Trend structure is bullish because price is above all major moving averages."
    elif price < ema20_val and price < ema50_val and price < ema200_val:
        technicals_score = 2
        technicals_summary += "Trend structure is weak because price is below major moving averages."
    else:
        technicals_summary += "Trend structure is mixed, so confirmation matters."

    if rsi_val > 70:
        technicals_summary += " RSI is elevated, which suggests overbought conditions."
    elif rsi_val < 30:
        technicals_summary += " RSI is depressed, which suggests oversold conditions."

    # Risk Management
    daily_returns = df["Close"].pct_change().dropna()
    volatility = float(daily_returns.std()) if not daily_returns.empty else 0

    if volatility < 0.02:
        risk_score = 4
        position_size = "Normal position sizing may be reasonable."
        risk_summary = "Volatility is relatively controlled for a stock."
    elif volatility < 0.04:
        risk_score = 3
        position_size = "Use moderate position sizing."
        risk_summary = "Volatility is moderate and requires discipline around entries."
    else:
        risk_score = 2
        position_size = "Use smaller position sizing."
        risk_summary = "Volatility is elevated, so risk needs tighter control."

    risk_summary += " Free data cannot fully map binary risks like surprise guidance, legal shocks, or sudden event risk."

    # Trading Plan
    if price > ema200_val and sentiment_score >= 50:
        trading_plan_score = 4
        trading_plan_summary = (
            "Core thesis favors trend continuation over the next 6-12 months, provided price holds above long-term support."
        )
    elif price > ema200_val:
        trading_plan_score = 3
        trading_plan_summary = (
            "Longer-term trend is still constructive, but softer sentiment means the thesis needs more confirmation."
        )
    else:
        trading_plan_score = 2
        trading_plan_summary = (
            "The long-term setup is less convincing because price is below the 200 EMA, so the 6-12 month thesis is weaker."
        )

    # Entry / Exit
    entry_zone_low = min(price, ema20_val, ema50_val)
    entry_zone_high = max(min(price, ema20_val), min(price, ema50_val))
    target_price = max(price * 1.15, resistance * 1.08)
    stop_loss = min(support, price * 0.92)

    entry_exit_score = 3
    entry_exit_summary = (
        f"Possible entry zone is {entry_zone_low:.2f} to {entry_zone_high:.2f}. "
        f"A 12-month target price is {target_price:.2f}. "
        f"A protective stop-loss is {stop_loss:.2f}."
    )

    # Strong Mindset
    if volatility >= 0.04:
        mindset_score = 3
        mindset_summary = (
            "Main emotional pitfall is panic during volatility spikes. Rules: reduce size, predefine exits, and never average down impulsively."
        )
    else:
        mindset_score = 4
        mindset_summary = (
            "Main emotional pitfall is overconfidence in a stable trend. Rules: do not chase, respect stops, and review the thesis before adding."
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
        "price": price,
        "rsi": rsi_val,
        "support": support,
        "resistance": resistance,
        "position_size": position_size,
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
# CHART
# -----------------------------
def build_chart(df, ticker):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.62, 0.18, 0.20]
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candles"
        ),
        row=1,
        col=1
    )

    fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"], mode="lines", name="EMA 9"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], mode="lines", name="EMA 20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], mode="lines", name="EMA 50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], mode="lines", name="EMA 200"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], mode="lines", name="VWAP"), row=1, col=1)

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume"
        ),
        row=2,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RSI"],
            mode="lines",
            name="RSI"
        ),
        row=3,
        col=1
    )

    fig.add_hline(y=70, line_dash="dash", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", row=3, col=1)

    fig.update_layout(
        title=f"{ticker} Price Chart",
        xaxis_rangeslider_visible=False,
        height=850,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=50, b=20)
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)

    return fig

# -----------------------------
# SCORE DISPLAY
# -----------------------------
def score_label(score):
    if score >= 4:
        return "Strong"
    if score >= 3:
        return "Average"
    return "Weak"

# -----------------------------
# UI
# -----------------------------
st.title("AI Stock Intelligence Dashboard")
st.write("Candles, volume, VWAP, RSI, news sentiment, and 6-pillar analysis using free data sources.")

top1, top2, top3 = st.columns([2, 1, 1])

with top1:
    ticker = st.text_input("Enter Ticker", "").strip().upper()

with top2:
    period = st.selectbox("Time Range", ["3mo", "6mo", "1y"], index=1)

with top3:
    interval = st.selectbox("Interval", ["1d", "1wk"], index=0)

if not ticker:
    st.info("Enter a ticker above to begin.")
    st.stop()

df, info, error = get_stock_data(ticker, period, interval)

if error:
    st.error(f"Data temporarily unavailable: {error}")
    st.info("Free Yahoo Finance data can get rate-limited. Wait a minute and try again.")
    st.stop()

news = get_news(ticker)
sentiment_score, sentiment_label = calculate_sentiment(news)
analysis = analyze(df, info or {}, ticker, sentiment_score)
ai_comments = generate_ai_commentary(
    ticker=ticker,
    price=analysis["price"],
    rsi_value=analysis["rsi"],
    overall_score=analysis["overall_score"],
    sentiment_label=sentiment_label,
    sentiment_score=sentiment_score
)

latest_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else latest_close
change = latest_close - prev_close
pct_change = (change / prev_close * 100) if prev_close != 0 else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Ticker", ticker)
m2.metric("Last Price", f"{latest_close:.2f}", f"{change:.2f} ({pct_change:.2f}%)")
m3.metric("Sentiment", f"{sentiment_score}/100")
m4.metric("Overall Score", f"{analysis['overall_score']} / 5")

st.plotly_chart(build_chart(df, ticker), use_container_width=True)

left, right = st.columns([1, 2])

with left:
    st.subheader("Sentiment Meter")
    st.progress(sentiment_score / 100)
    st.markdown(f"**Headline Sentiment:** {sentiment_label}")
    st.caption("This free version uses news-headline sentiment, not live Reddit/X sentiment.")

    st.subheader("AI Comments")
    for c in ai_comments:
        st.markdown(f"- {c}")

with right:
    st.subheader("Latest News")
    if news:
        for item in news[:8]:
            with st.container(border=True):
                st.markdown(f"**{item['title']}**")
                if item["published"]:
                    st.caption(item["published"])
                st.markdown(f"[Open article]({item['link']})")
    else:
        st.write("No news available right now.")

st.subheader("Comprehensive 6-Pillar Trading Analysis")

with st.container(border=True):
    st.markdown("**1. Fundamentals**")
    st.write(analysis["fundamentals_summary"])

with st.container(border=True):
    st.markdown("**2. Technicals**")
    st.write(analysis["technicals_summary"])

with st.container(border=True):
    st.markdown("**3. Risk Management**")
    st.write(analysis["risk_summary"])
    st.caption(analysis["position_size"])

with st.container(border=True):
    st.markdown("**4. Trading Plan**")
    st.write(analysis["trading_plan_summary"])

with st.container(border=True):
    st.markdown("**5. Entry/Exit Strategy**")
    st.write(analysis["entry_exit_summary"])

with st.container(border=True):
    st.markdown("**6. Strong Mindset**")
    st.write(analysis["mindset_summary"])

st.subheader("Consolidated Rating Table")

rating_df = pd.DataFrame({
    "Pillar": list(analysis["scores"].keys()),
    "Score": list(analysis["scores"].values()),
    "Rating": [score_label(v) for v in analysis["scores"].values()]
})

def highlight_scores(row):
    score = row["Score"]
    if score >= 4:
        color = "#123524"
    elif score >= 3:
        color = "#3a2f0b"
    else:
        color = "#4b1f1f"
    return [f"background-color: {color}; color: white"] * len(row)

styled_df = rating_df.style.apply(highlight_scores, axis=1)

st.dataframe(styled_df, use_container_width=True)
st.success(f"Overall Score: {analysis['overall_score']} / 5")
