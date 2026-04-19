import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import csv
import json

st.set_page_config(
    page_title="Utpal Trading Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS — TradingView-style dark theme
# ─────────────────────────────────────────
st.markdown("""
<style>
body, .stApp { background-color: #131722; color: #d1d4dc; font-family: 'Trebuchet MS', sans-serif; }
.stTextInput > div > input { background-color: #1e222d; color: #d1d4dc; border: 1px solid #363a45; border-radius: 4px; }
.stSelectbox > div > div { background-color: #1e222d; color: #d1d4dc; }
.stButton > button { background-color: #2962ff; color: white; border: none; border-radius: 4px; padding: 6px 16px; font-weight: bold; }
.stButton > button:hover { background-color: #1e53e5; }
.metric-card { background: #1e222d; border: 1px solid #363a45; border-radius: 6px; padding: 12px 16px; margin: 4px; }
.pillar-card { background: #1e222d; border-left: 4px solid #2962ff; border-radius: 4px; padding: 14px; margin-bottom: 10px; }
.pillar-card.strong { border-left-color: #26a69a; }
.pillar-card.weak { border-left-color: #ef5350; }
.pillar-card.avg { border-left-color: #f5c842; }
.number-chip { background: #2a2e39; border-radius: 3px; padding: 2px 7px; font-size: 12px; margin-right: 4px; color: #d1d4dc; display: inline-block; }
.green { color: #26a69a; font-weight: bold; }
.red { color: #ef5350; font-weight: bold; }
.yellow { color: #f5c842; font-weight: bold; }
h1, h2, h3 { color: #d1d4dc; }
div[data-testid="stMetric"] { background: #1e222d; border: 1px solid #363a45; border-radius: 6px; padding: 10px 14px; }
div[data-testid="stMetricLabel"] { color: #787b86; font-size: 11px; }
div[data-testid="stMetricValue"] { color: #d1d4dc; font-size: 22px; font-weight: bold; }
.sidebar .sidebar-content { background: #1e222d; }
[data-testid="stSidebar"] { background-color: #1e222d; border-right: 1px solid #363a45; }
.risk-high { background: rgba(239,83,80,0.15); border: 1px solid #ef5350; border-radius: 6px; padding: 10px 14px; }
.risk-med  { background: rgba(245,200,66,0.10); border: 1px solid #f5c842; border-radius: 6px; padding: 10px 14px; }
.risk-low  { background: rgba(38,166,154,0.10); border: 1px solid #26a69a; border-radius: 6px; padding: 10px 14px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# PORTFOLIO CONFIG — uses .NE tickers (NEO Exchange Canada)
# ─────────────────────────────────────────
DEFAULT_PORTFOLIO = {
    "AVGO.NE": {"shares": 1005, "avg_cost": 13.4353, "currency": "CAD", "target": 20.0},
    "META.NE": {"shares": 417,  "avg_cost": 32.59,   "currency": "CAD", "target": 45.0},
}

DEFAULT_WATCHLIST = [
    "AVGO", "META", "NVDA", "AMD", "AMZN", "MSFT", "GOOGL",
    "DOCN", "NET", "ALAB", "COHR", "MU", "INTC", "NFLX", "CLS"
]

JOURNAL_FILE = "trade_journal.csv"
JOURNAL_COLS = [
    "date", "ticker", "direction", "asset_type", "entry", "stop", "target",
    "size", "thesis", "timeframe", "alignment_score", "catalyst_state",
    "options_setup", "result", "pnl", "notes", "mistakes"
]

# ─────────────────────────────────────────
# INDICATOR FUNCTIONS
# ─────────────────────────────────────────
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift(1))
    low_close  = np.abs(df["Low"]  - df["Close"].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()

def bollinger_bands(series, period=20, std_dev=2):
    mid   = series.rolling(period).mean()
    sigma = series.rolling(period).std()
    return mid + std_dev * sigma, mid, mid - std_dev * sigma

def supertrend(df, period=10, multiplier=3):
    data = df.copy()
    hl2  = (data["High"] + data["Low"]) / 2
    data["ATR"] = atr(data, period)
    ub = hl2 + multiplier * data["ATR"]
    lb = hl2 - multiplier * data["ATR"]
    final_ub = ub.copy()
    final_lb = lb.copy()
    for i in range(1, len(data)):
        final_ub.iloc[i] = min(ub.iloc[i], final_ub.iloc[i-1]) if data["Close"].iloc[i-1] <= final_ub.iloc[i-1] else ub.iloc[i]
        final_lb.iloc[i] = max(lb.iloc[i], final_lb.iloc[i-1]) if data["Close"].iloc[i-1] >= final_lb.iloc[i-1] else lb.iloc[i]
    trend     = pd.Series(index=data.index, dtype="float64")
    direction = pd.Series(index=data.index, dtype="object")
    for i in range(len(data)):
        if i == 0:
            trend.iloc[i]     = final_lb.iloc[i]
            direction.iloc[i] = "up"
            continue
        if trend.iloc[i-1] == final_ub.iloc[i-1]:
            if data["Close"].iloc[i] <= final_ub.iloc[i]:
                trend.iloc[i]     = final_ub.iloc[i]
                direction.iloc[i] = "down"
            else:
                trend.iloc[i]     = final_lb.iloc[i]
                direction.iloc[i] = "up"
        else:
            if data["Close"].iloc[i] >= final_lb.iloc[i]:
                trend.iloc[i]     = final_lb.iloc[i]
                direction.iloc[i] = "up"
            else:
                trend.iloc[i]     = final_ub.iloc[i]
                direction.iloc[i] = "down"
    data["Supertrend"]          = trend
    data["SupertrendDirection"] = direction
    return data["Supertrend"], data["SupertrendDirection"]

# ─────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────
INTERVAL_MAP = {
    "1m":  "7d",
    "5m":  "30d",
    "15m": "60d",
    "1h":  "730d",
    "4h":  "730d",
    "1d":  "2y",
    "1wk": "5y",
}

@st.cache_data(ttl=300)
def get_stock_data(ticker, interval="1d"):
    try:
        period = INTERVAL_MAP.get(interval, "2y")
        stock  = yf.Ticker(ticker)
        df     = stock.history(period=period, interval=interval, auto_adjust=False)
        if df.empty:
            return None, None, f"No data for {ticker}"
        df = df.dropna().copy()
        df["EMA9"]    = ema(df["Close"], 9)
        df["EMA20"]   = ema(df["Close"], 20)
        df["EMA50"]   = ema(df["Close"], 50)
        df["EMA200"]  = ema(df["Close"], 200)
        df["RSI"]     = rsi(df["Close"])
        df["VWAP"]    = vwap(df)
        df["ATR"]     = atr(df, 14)
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd(df["Close"])
        df["BB_Upper"], df["BB_Mid"], df["BB_Lower"]   = bollinger_bands(df["Close"])
        df["Supertrend"], df["SupertrendDir"]          = supertrend(df, 10, 3)
        try:
            info = stock.info
        except Exception:
            info = {}
        return df, info, None
    except Exception as e:
        return None, None, str(e)

# ─────────────────────────────────────────
# PORTFOLIO TICKER PRICE — uses the exact ticker (e.g. AVGO.NE)
# ─────────────────────────────────────────
@st.cache_data(ttl=300)
def get_portfolio_price(ticker):
    """Fetch current price using the EXACT ticker symbol provided."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="5d", interval="1d", auto_adjust=False)
        if df is not None and not df.empty:
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
    return None

# ─────────────────────────────────────────
# NEWS + SENTIMENT
# ─────────────────────────────────────────
@st.cache_data(ttl=1800)
def get_news(ticker):
    try:
        base = ticker.split(".")[0]
        url  = f"https://news.google.com/rss/search?q={base}+stock+earnings"
        feed = feedparser.parse(url)
        return [{"title": e.title, "link": e.link, "published": getattr(e, "published", "")}
                for e in feed.entries[:12]]
    except Exception:
        return []

def calculate_sentiment(news_items):
    bull_words = {"beat","beats","surge","strong","upgrade","record","growth","profit","soar","buy","outperform","expands","positive","rebound","bullish","momentum","raises","raised","tops"}
    bear_words = {"miss","misses","drop","downgrade","warning","weak","fall","lawsuit","loss","decline","probe","investigation","negative","concern","bearish","cut","cuts","risks","below"}
    score = 0
    hits  = 0
    for item in news_items:
        t    = item["title"].lower()
        bull = sum(1 for w in bull_words if w in t)
        bear = sum(1 for w in bear_words if w in t)
        score += bull - bear
        hits  += bull + bear
    norm  = 50 if hits == 0 else int(max(0, min(100, 50 + max(-40, min(40, score * 8)))))
    label = "Bullish" if norm >= 65 else ("Bearish" if norm <= 35 else "Neutral")
    return norm, label

# ─────────────────────────────────────────
# SUPPORT / RESISTANCE
# ─────────────────────────────────────────
def get_levels(df):
    s20  = float(df["Low"].tail(20).min())
    s50  = float(df["Low"].tail(50).min())  if len(df) >= 50  else s20
    r20  = float(df["High"].tail(20).max())
    r50  = float(df["High"].tail(50).max()) if len(df) >= 50  else r20
    s200 = float(df["Low"].tail(200).min()) if len(df) >= 200 else s50
    r200 = float(df["High"].tail(200).max()) if len(df) >= 200 else r50
    return s20, s50, s200, r20, r50, r200

# ─────────────────────────────────────────
# CATALYST / EARNINGS AWARENESS
# ─────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_catalyst_data(ticker):
    """Fetch earnings date, ex-div date, analyst target from yfinance."""
    result = {
        "earnings_date": None,
        "days_to_earnings": None,
        "ex_div_date": None,
        "analyst_target": None,
        "event_risk": "Low",
        "risk_label": "🟢 Low Risk",
        "risk_css": "risk-low",
    }
    try:
        stock = yf.Ticker(ticker)
        cal   = stock.calendar
        info  = stock.info or {}

        # Earnings date
        if cal is not None and not cal.empty:
            try:
                if hasattr(cal, "columns") and "Earnings Date" in cal.columns:
                    ed_raw = cal["Earnings Date"].iloc[0]
                elif isinstance(cal, dict) and "Earnings Date" in cal:
                    ed_raw = cal["Earnings Date"]
                    if isinstance(ed_raw, list):
                        ed_raw = ed_raw[0]
                else:
                    ed_raw = None

                if ed_raw is not None:
                    if hasattr(ed_raw, "date"):
                        ed = ed_raw.date()
                    else:
                        ed = pd.Timestamp(ed_raw).date()
                    result["earnings_date"] = ed
                    dte = (ed - datetime.now().date()).days
                    result["days_to_earnings"] = dte
                    if dte <= 5:
                        result["event_risk"]  = "High"
                        result["risk_label"]  = "🔴 High Risk — Earnings ≤5 Days"
                        result["risk_css"]    = "risk-high"
                    elif dte <= 14:
                        result["event_risk"]  = "Moderate"
                        result["risk_label"]  = "🟡 Moderate Risk — Earnings 6–14 Days"
                        result["risk_css"]    = "risk-med"
            except Exception:
                pass

        # Ex-dividend date
        try:
            ex_div = info.get("exDividendDate")
            if ex_div:
                result["ex_div_date"] = pd.Timestamp(ex_div, unit="s").date()
        except Exception:
            pass

        # Analyst target
        result["analyst_target"] = info.get("targetMeanPrice")

    except Exception:
        pass
    return result

# ─────────────────────────────────────────
# OPTIONS ANALYSIS
# ─────────────────────────────────────────
@st.cache_data(ttl=1800)
def get_options_data(ticker):
    """Fetch IV and options metadata; fail gracefully."""
    result = {
        "iv": None, "iv_rank": None, "next_exp": None,
        "has_options": False, "error": None,
    }
    try:
        stock = yf.Ticker(ticker)
        exps  = stock.options
        if not exps:
            result["error"] = "No options chain available"
            return result
        result["has_options"] = True
        result["next_exp"]    = exps[0]

        chain = stock.option_chain(exps[0])
        calls = chain.calls
        if "impliedVolatility" in calls.columns and not calls.empty:
            result["iv"] = float(calls["impliedVolatility"].dropna().median()) * 100
    except Exception as e:
        result["error"] = str(e)
    return result

def options_recommendation(trend_bias, event_risk, iv, rsi_val, atr_pct):
    """Return a recommended options vehicle based on available data."""
    if event_risk == "High":
        return "⚠️ Avoid / Wait", "Earnings within 5 days. IV is inflated — selling premium might work but direction unpredictable. Best to wait until after the event."

    if iv is None:
        # No IV data — fallback to stock recommendation
        if trend_bias == "Bullish" and rsi_val < 65:
            return "📈 Stock Only / Calls", "No IV data available. Trend is bullish and RSI not extended — stock long or ATM calls acceptable."
        elif trend_bias == "Bearish":
            return "📉 Puts / Stock Short", "No IV data available. Trend is bearish — puts or short stock are directionally aligned."
        else:
            return "⏳ Wait", "Trend is mixed and IV data unavailable. Insufficient data to recommend a vehicle."

    if iv > 60 and event_risk in ("Low", "Moderate"):
        if trend_bias == "Bullish":
            return "🐂 Bull Call Spread", f"IV at {iv:.1f}% is elevated. Spread caps premium paid while expressing bullish view. Better than naked calls here."
        elif trend_bias == "Bearish":
            return "🐻 Bear Put Spread", f"IV at {iv:.1f}% is elevated. Spread reduces premium cost for a bearish bet."
        else:
            return "⏳ Wait", f"IV elevated ({iv:.1f}%) but trend is unclear. Not enough directional conviction to trade options."

    if iv <= 30:
        if trend_bias == "Bullish" and rsi_val < 65:
            return "📈 Calls (ATM/Slight ITM)", f"Low IV ({iv:.1f}%) — options are cheap. ATM calls give good leverage on a bullish move."
        elif trend_bias == "Bearish":
            return "📉 Puts (ATM)", f"Low IV ({iv:.1f}%) — puts are relatively cheap. Favourable for directional bearish bets."

    if trend_bias == "Bullish" and rsi_val < 65 and atr_pct < 3:
        return "📈 Calls", f"Moderate IV ({iv:.1f}%), bullish trend, RSI healthy. Calls aligned with trend."
    elif trend_bias == "Bearish" and rsi_val > 35:
        return "📉 Puts", f"Moderate IV ({iv:.1f}%), bearish trend. Puts aligned."
    elif atr_pct > 4:
        return "🔄 Stock Only", f"High volatility ({atr_pct:.1f}% ATR) makes options expensive relative to move needed. Stock preferred."

    return "⏳ Wait", "Signals not clear enough for a high-conviction options setup."

# ─────────────────────────────────────────
# MULTI-TIMEFRAME CONFIRMATION
# ─────────────────────────────────────────
@st.cache_data(ttl=300)
def get_mtf_data(ticker):
    """Build signals across 6 timeframes and produce alignment score."""
    timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
    rows = []
    bullish_count = 0
    total_tfs = 0

    for tf in timeframes:
        df_tf, _, err = get_stock_data(ticker, tf)
        if err or df_tf is None or len(df_tf) < 20:
            rows.append({"TF": tf, "Trend": "—", "EMA9": "—", "EMA20": "—",
                         "RSI": "—", "MACD": "—", "ST": "—", "Verdict": "⬜ No Data"})
            continue

        lat = df_tf.iloc[-1]
        p   = float(lat["Close"])
        e9  = float(lat["EMA9"])
        e20 = float(lat["EMA20"])
        r   = float(lat["RSI"])
        mc  = float(lat["MACD"])
        ms  = float(lat["MACD_Signal"])
        sd  = str(lat["SupertrendDir"])
        vw  = float(lat["VWAP"]) if "VWAP" in df_tf.columns else None

        bull_signals = 0
        bear_signals = 0

        ema9_lbl  = "▲" if p > e9  else "▼"
        ema20_lbl = "▲" if p > e20 else "▼"
        macd_lbl  = "▲" if mc > ms else "▼"
        st_lbl    = "▲" if sd == "up" else "▼"

        if p > e9:   bull_signals += 1
        else:        bear_signals += 1
        if p > e20:  bull_signals += 1
        else:        bear_signals += 1
        if mc > ms:  bull_signals += 1
        else:        bear_signals += 1
        if sd == "up":  bull_signals += 1
        else:           bear_signals += 1
        if r > 50:   bull_signals += 1
        else:        bear_signals += 1
        if vw and p > vw:
            bull_signals += 1
        elif vw and p < vw:
            bear_signals += 1

        total = bull_signals + bear_signals
        if total == 0: total = 1
        bull_pct = bull_signals / total

        if bull_pct >= 0.75:
            verdict = "🟢 Bullish"
            bullish_count += 1
        elif bull_pct >= 0.5:
            verdict = "🟡 Mild Bull"
            bullish_count += 0.5
        elif bull_pct >= 0.25:
            verdict = "🟠 Mild Bear"
        else:
            verdict = "🔴 Bearish"

        total_tfs += 1
        rsi_lbl = f"{r:.0f} {'↑' if r > 55 else '↓' if r < 45 else '→'}"

        rows.append({
            "TF": tf, "EMA9": ema9_lbl, "EMA20": ema20_lbl,
            "RSI": rsi_lbl, "MACD": macd_lbl, "ST": st_lbl, "Verdict": verdict,
        })

    align_score = int((bullish_count / max(total_tfs, 1)) * 100)

    if align_score >= 75:
        primary_bias   = "Strong Bullish"
        confirmation   = "Multiple timeframes confirm uptrend"
        trigger        = "Enter on 15m/1h pullback to EMA9 or EMA20"
        conflict_note  = "Watch for 1d RSI if overbought"
    elif align_score >= 55:
        primary_bias   = "Mild Bullish"
        confirmation   = "Short timeframes lean bull, higher TFs mixed"
        trigger        = "Wait for 1h close above EMA20 before entry"
        conflict_note  = "Some TF conflict — size down, tighter stop"
    elif align_score >= 45:
        primary_bias   = "Neutral / Choppy"
        confirmation   = "No clear multi-TF alignment"
        trigger        = "Avoid — wait for resolution"
        conflict_note  = "Significant TF conflict — high whipsaw risk"
    elif align_score >= 25:
        primary_bias   = "Mild Bearish"
        confirmation   = "Most TFs lean bearish"
        trigger        = "Avoid longs. Watch for short entry on bounces"
        conflict_note  = "1d may still be transitioning"
    else:
        primary_bias   = "Strong Bearish"
        confirmation   = "All timeframes aligned bearish"
        trigger        = "Short bias. Enter on failed bounces to EMA9/20"
        conflict_note  = "Confirm with volume — no catching falling knives"

    return {
        "rows": rows,
        "alignment_score": align_score,
        "primary_bias": primary_bias,
        "confirmation": confirmation,
        "trigger": trigger,
        "conflict_note": conflict_note,
    }

# ─────────────────────────────────────────
# POSITION SIZING CALCULATOR
# ─────────────────────────────────────────
def calc_position_size(account_size, cash_avail, risk_pct, entry, stop, target1, target2, asset_type, atr_val=None):
    if entry <= 0 or stop <= 0 or stop >= entry:
        return None, "⚠️ Stop must be below entry and both > 0."

    max_risk_dollars = account_size * (risk_pct / 100)
    risk_per_unit    = entry - stop
    units            = max_risk_dollars / risk_per_unit if risk_per_unit > 0 else 0

    if asset_type == "Option":
        units = int(units / 100)  # 1 contract = 100 shares
        if units < 1:
            units = 1
        pos_value = units * 100 * entry
        label_unit = "contracts"
    else:
        units     = int(units)
        pos_value = units * entry
        label_unit = "shares"

    rr1 = (target1 - entry) / risk_per_unit if risk_per_unit > 0 and target1 > entry else 0
    rr2 = (target2 - entry) / risk_per_unit if risk_per_unit > 0 and target2 > entry else 0

    warnings = []
    if pos_value > cash_avail:
        warnings.append(f"⚠️ Position value ${pos_value:,.2f} exceeds available cash ${cash_avail:,.2f}. Scale down.")
    if atr_val:
        atr_pct_stop = (risk_per_unit / atr_val)
        if atr_pct_stop < 0.5:
            warnings.append(f"⚠️ Stop is very tight ({atr_pct_stop:.1f}x ATR). High chance of stop-out on normal noise.")
        elif atr_pct_stop > 3:
            warnings.append(f"⚠️ Stop is wide ({atr_pct_stop:.1f}x ATR). Risk per trade may be acceptable but position value is small.")
    if rr1 > 0 and rr1 < 1.5:
        warnings.append(f"⚠️ R:R to Target 1 is only {rr1:.2f}x — below the minimum 1.5x threshold.")

    return {
        "max_risk_dollars": max_risk_dollars,
        "units": units,
        "label_unit": label_unit,
        "pos_value": pos_value,
        "risk_per_unit": risk_per_unit,
        "rr1": rr1,
        "rr2": rr2,
    }, "\n".join(warnings) if warnings else None

# ─────────────────────────────────────────
# JOURNAL HELPERS
# ─────────────────────────────────────────
def load_journal():
    if not os.path.exists(JOURNAL_FILE):
        return pd.DataFrame(columns=JOURNAL_COLS)
    try:
        df = pd.read_csv(JOURNAL_FILE)
        for col in JOURNAL_COLS:
            if col not in df.columns:
                df[col] = ""
        return df
    except Exception:
        return pd.DataFrame(columns=JOURNAL_COLS)

def save_trade(trade_dict):
    exists = os.path.exists(JOURNAL_FILE)
    with open(JOURNAL_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_COLS)
        if not exists:
            writer.writeheader()
        row = {col: trade_dict.get(col, "") for col in JOURNAL_COLS}
        writer.writerow(row)

def journal_analytics(df):
    if df.empty or "result" not in df.columns:
        return {}
    df = df.copy()
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0)
    df["alignment_score"] = pd.to_numeric(df["alignment_score"], errors="coerce").fillna(0)

    total   = len(df)
    wins    = df[df["result"].str.lower() == "win"]
    losses  = df[df["result"].str.lower() == "loss"]
    win_rate = len(wins) / total * 100 if total > 0 else 0
    avg_win  = wins["pnl"].mean() if not wins.empty else 0
    avg_loss = losses["pnl"].mean() if not losses.empty else 0
    total_pnl = df["pnl"].sum()
    gross_profit = wins["pnl"].sum()
    gross_loss   = abs(losses["pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

    insights = []
    # Earnings proximity insight
    near_earn = df[df["catalyst_state"].str.contains("High", na=False)]
    far_earn  = df[~df["catalyst_state"].str.contains("High", na=False)]
    if len(near_earn) >= 2 and len(far_earn) >= 2:
        wr_near = len(near_earn[near_earn["result"].str.lower() == "win"]) / len(near_earn) * 100
        wr_far  = len(far_earn[far_earn["result"].str.lower() == "win"]) / len(far_earn) * 100
        if wr_near < wr_far - 10:
            insights.append(f"📉 Win rate drops near earnings: {wr_near:.0f}% (near earnings) vs {wr_far:.0f}% (normal). Avoid trading high-risk earnings windows.")

    # Alignment score insight
    high_align = df[df["alignment_score"] >= 70]
    low_align  = df[df["alignment_score"] < 50]
    if len(high_align) >= 2 and len(low_align) >= 2:
        wr_h = len(high_align[high_align["result"].str.lower() == "win"]) / len(high_align) * 100
        wr_l = len(low_align[low_align["result"].str.lower() == "win"]) / len(low_align) * 100
        if wr_h > wr_l + 10:
            insights.append(f"✅ Trades with alignment ≥70 win {wr_h:.0f}% vs {wr_l:.0f}% for <50. Prioritize high-alignment setups.")

    return {
        "total": total, "win_rate": win_rate, "avg_win": avg_win,
        "avg_loss": avg_loss, "total_pnl": total_pnl,
        "profit_factor": profit_factor, "expectancy": expectancy, "insights": insights,
    }

# ─────────────────────────────────────────
# 6-PILLAR ANALYSIS
# ─────────────────────────────────────────
def fmt_large(n):
    if n is None: return "N/A"
    if n >= 1e12: return f"${n/1e12:.2f}T"
    if n >= 1e9:  return f"${n/1e9:.2f}B"
    if n >= 1e6:  return f"${n/1e6:.2f}M"
    return f"${n:,.0f}"

def analyze(df, info, ticker, sentiment_score):
    latest  = df.iloc[-1]
    price   = float(latest["Close"])
    ema9    = float(latest["EMA9"])
    ema20   = float(latest["EMA20"])
    ema50   = float(latest["EMA50"])
    ema200  = float(latest["EMA200"])
    rsi_val = float(latest["RSI"])
    vwap_v  = float(latest["VWAP"])
    st_val  = float(latest["Supertrend"])
    st_dir  = str(latest["SupertrendDir"])
    atr_val = float(latest["ATR"]) if not np.isnan(float(latest["ATR"])) else price * 0.02
    macd_v  = float(latest["MACD"])
    macd_s  = float(latest["MACD_Signal"])
    bb_u    = float(latest["BB_Upper"])
    bb_l    = float(latest["BB_Lower"])
    bb_m    = float(latest["BB_Mid"])

    revenue      = info.get("totalRevenue")
    rev_growth   = info.get("revenueGrowth")
    gross_margin = info.get("grossMargins")
    op_margin    = info.get("operatingMargins")
    net_margin   = info.get("profitMargins")
    pe           = info.get("trailingPE")
    fwd_pe       = info.get("forwardPE")
    peg          = info.get("pegRatio")
    eps          = info.get("trailingEps")
    debt         = info.get("totalDebt")
    cash         = info.get("totalCash")
    fcf          = info.get("freeCashflow")
    roe          = info.get("returnOnEquity")
    long_name    = info.get("longName", ticker)
    sector       = info.get("sector", "")
    market_cap   = info.get("marketCap")

    s20, s50, s200, r20, r50, r200 = get_levels(df)

    # ── PILLAR 1: FUNDAMENTALS ─────────────────────────────────
    fund_score   = 3
    fund_details = []
    fund_issues  = []

    if revenue:
        fund_details.append(f"Revenue: **{fmt_large(revenue)}**")
    if rev_growth is not None:
        pct = rev_growth * 100
        if pct > 15:
            fund_details.append(f"Revenue Growth: **{pct:.1f}%** ✅")
            fund_score += 0.5
        elif pct > 0:
            fund_details.append(f"Revenue Growth: **{pct:.1f}%** (moderate)")
        else:
            fund_details.append(f"Revenue Growth: **{pct:.1f}%** ⚠️")
            fund_issues.append(f"Revenue is shrinking ({pct:.1f}%)")
            fund_score -= 0.5
    if gross_margin is not None:
        gm  = gross_margin * 100
        lbl = "✅ high" if gm > 50 else ("moderate" if gm > 30 else "⚠️ low")
        fund_details.append(f"Gross Margin: **{gm:.1f}%** ({lbl})")
        if gm < 30:
            fund_issues.append(f"Low gross margin {gm:.1f}% = thin pricing power")
            fund_score -= 0.5
    if net_margin is not None:
        nm  = net_margin * 100
        lbl = "✅" if nm > 15 else ("⚠️" if nm < 0 else "")
        fund_details.append(f"Net Margin: **{nm:.1f}%** {lbl}")
        if nm < 0:
            fund_issues.append(f"Company is losing money (net margin {nm:.1f}%)")
            fund_score -= 1
    if pe is not None:
        lbl = "elevated" if pe > 40 else ("reasonable" if pe < 20 else "moderate")
        fund_details.append(f"P/E: **{pe:.1f}x** ({lbl})")
        if pe > 60:
            fund_issues.append(f"P/E of {pe:.1f}x is very high — priced for perfection")
            fund_score -= 0.5
    if fwd_pe is not None:
        fund_details.append(f"Fwd P/E: **{fwd_pe:.1f}x**")
    if peg is not None:
        lbl = "✅ cheap vs growth" if peg < 1.5 else ("⚠️ expensive vs growth" if peg > 3 else "fair")
        fund_details.append(f"PEG: **{peg:.2f}** ({lbl})")
        if peg > 3:
            fund_issues.append(f"PEG of {peg:.2f} means you're paying a lot for expected growth")
    if eps is not None:
        fund_details.append(f"EPS (TTM): **${eps:.2f}**")
    if cash is not None and debt is not None:
        net_cash = cash - debt
        if net_cash > 0:
            fund_details.append(f"Net Cash: **{fmt_large(net_cash)}** ✅")
            fund_score += 0.5
        else:
            fund_details.append(f"Net Debt: **{fmt_large(abs(net_cash))}** ⚠️")
            fund_issues.append(f"Debt exceeds cash by {fmt_large(abs(net_cash))}")
            fund_score -= 0.5
    if fcf is not None:
        lbl = "✅ positive" if fcf > 0 else "⚠️ negative"
        fund_details.append(f"Free Cash Flow: **{fmt_large(fcf)}** ({lbl})")
        if fcf < 0:
            fund_issues.append("Negative free cash flow — burns more cash than it generates")
            fund_score -= 0.5
    if roe is not None:
        roe_pct = roe * 100
        lbl = "✅ strong" if roe_pct > 15 else ("⚠️ weak" if roe_pct < 5 else "moderate")
        fund_details.append(f"ROE: **{roe_pct:.1f}%** ({lbl})")

    fund_score    = max(1, min(5, round(fund_score)))
    fund_why_weak = ("**Why fundamentals are weak:**\n" + "\n".join(f"- {x}" for x in fund_issues)) if fund_issues else ""
    fund_summary  = "\n".join(f"- {d}" for d in fund_details) if fund_details else "Fundamental data not available for this ticker."

    # ── PILLAR 2: TECHNICALS ──────────────────────────────────
    tech_score   = 3
    tech_details = []
    tech_issues  = []
    trend_bias   = "Neutral"

    tech_details.append(f"Price: **${price:.2f}**")
    tech_details.append(f"EMA9: **${ema9:.2f}** | EMA20: **${ema20:.2f}** | EMA50: **${ema50:.2f}** | EMA200: **${ema200:.2f}**")
    tech_details.append(f"RSI (14): **{rsi_val:.1f}**")
    tech_details.append(f"MACD: **{macd_v:.3f}** vs Signal: **{macd_s:.3f}**")
    tech_details.append(f"VWAP: **${vwap_v:.2f}**")
    tech_details.append(f"ATR (14): **${atr_val:.2f}** (daily expected move)")
    tech_details.append(f"Bollinger Bands: **${bb_l:.2f} — ${bb_m:.2f} — ${bb_u:.2f}**")
    tech_details.append(f"Supertrend: **{st_dir.upper()}** at **${st_val:.2f}**")

    bullish_stack = price > ema9 > ema20 > ema50 and price > ema200 and price > st_val
    bearish_stack = price < ema9 < ema20 < ema50 and price < ema200 and price < st_val

    if bullish_stack:
        tech_score = 5; trend_bias = "Bullish"
        tech_details.append("✅ **Perfect bull stack**: Price > EMA9 > EMA20 > EMA50 > EMA200. Supertrend UP.")
    elif price > ema20 and price > ema50 and price > ema200:
        tech_score = 4; trend_bias = "Bullish"
        tech_details.append("✅ **Bullish**: Price above all key MAs. Not full perfect stack.")
    elif bearish_stack:
        tech_score = 1; trend_bias = "Bearish"
        tech_issues.append("Price below EMA9 < EMA20 < EMA50 < EMA200. Supertrend DOWN.")
    elif price < ema50 and price < ema200:
        tech_score = 2; trend_bias = "Bearish"
        tech_issues.append(f"Price ${price:.2f} is below EMA50 (${ema50:.2f}) and EMA200 (${ema200:.2f})")

    if rsi_val > 70:
        tech_details.append(f"⚠️ **RSI {rsi_val:.1f} is overbought** — short-term pullback risk")
    elif rsi_val < 30:
        tech_details.append(f"⚠️ **RSI {rsi_val:.1f} is oversold** — potential bounce zone")
    else:
        tech_details.append(f"✅ RSI {rsi_val:.1f} is in healthy range")

    if macd_v > macd_s:
        tech_details.append("✅ MACD above signal — bullish momentum")
    else:
        tech_details.append("⚠️ MACD below signal — bearish momentum")

    vwap_pct = ((price - vwap_v) / vwap_v) * 100
    if price > vwap_v:
        tech_details.append(f"✅ Price {vwap_pct:+.1f}% above VWAP — buyers in control")
    else:
        tech_details.append(f"⚠️ Price {vwap_pct:+.1f}% below VWAP — sellers in control")

    bb_pos = ((price - bb_l) / (bb_u - bb_l)) * 100 if (bb_u - bb_l) > 0 else 50
    tech_details.append(f"BB position: **{bb_pos:.0f}%** of band ({'upper zone' if bb_pos > 80 else 'lower zone' if bb_pos < 20 else 'mid zone'})")

    tech_score    = max(1, min(5, tech_score))
    tech_why_weak = ("**Why technicals are weak:**\n" + "\n".join(f"- {x}" for x in tech_issues)) if tech_issues else ""
    tech_summary  = "\n".join(f"- {d}" for d in tech_details)

    # ── PILLAR 3: RISK MANAGEMENT ─────────────────────────────
    daily_ret    = df["Close"].pct_change().dropna()
    vol          = float(daily_ret.std()) if not daily_ret.empty else 0
    vol_pct      = vol * 100
    max_drawdown = float(((df["Close"] / df["Close"].cummax()) - 1).min()) * 100

    risk_details = [
        f"Daily Volatility (std): **{vol_pct:.2f}%**",
        f"ATR-based daily move: **±${atr_val:.2f}** (**{atr_val/price*100:.1f}%** of price)",
        f"Max Drawdown (period): **{max_drawdown:.1f}%**",
        f"Implied Stop Loss level: **${price - atr_val * 1.5:.2f}** (1.5x ATR below price)",
    ]

    if vol < 0.01:
        risk_score   = 4
        risk_summary = f"Low volatility ({vol_pct:.2f}%/day). Supports cleaner entries and stops."
        pos_size     = "Normal sizing appropriate."
    elif vol < 0.025:
        risk_score   = 3
        risk_summary = f"Moderate volatility ({vol_pct:.2f}%/day). Respect support levels."
        pos_size     = "Moderate sizing — use ATR-based stops."
    else:
        risk_score   = 2
        risk_summary = f"High volatility ({vol_pct:.2f}%/day). Widen stops or reduce size."
        pos_size     = f"Reduce position size — daily swings of ~${atr_val:.2f} expected."

    if max_drawdown < -40:
        risk_details.append(f"⚠️ Max drawdown of {max_drawdown:.1f}% means this stock can cut in half")
        risk_score = max(1, risk_score - 1)

    risk_details.append(f"Suggested position size: **{pos_size}**")
    risk_details.append("Risk note: Earnings, macro events not captured by historical volatility.")
    risk_why_weak = "" if risk_score >= 3 else f"**Why risk is elevated:** Volatility of {vol_pct:.2f}%/day with max drawdown {max_drawdown:.1f}%"
    risk_full     = "\n".join(f"- {d}" for d in risk_details)

    # ── PILLAR 4: TRADING PLAN ────────────────────────────────
    if trend_bias == "Bullish" and sentiment_score >= 50:
        plan_score   = 4
        plan_summary = f"Bullish trend + positive sentiment ({sentiment_score}/100) = constructive setup. Buy pullbacks to EMA20 (${ema20:.2f}) or EMA50 (${ema50:.2f}). Hold while Supertrend is UP and price stays above EMA50."
    elif trend_bias == "Bullish":
        plan_score   = 3
        plan_summary = f"Trend is bullish but sentiment is mixed ({sentiment_score}/100). Wait for pullbacks. EMA20 (${ema20:.2f}) is key support to watch."
    elif trend_bias == "Bearish":
        plan_score   = 2
        plan_summary = f"Bearish trend (price below EMA50 ${ema50:.2f}). Avoid longs unless price reclaims EMA50. Better to wait for structure shift."
    else:
        plan_score   = 3
        plan_summary = f"Mixed signals — price is between key MAs. EMA50 (${ema50:.2f}) and EMA200 (${ema200:.2f}) are the critical levels to watch for confirmation."

    # ── PILLAR 5: ENTRY / EXIT ────────────────────────────────
    entry_low    = min(ema20, ema50) if trend_bias == "Bullish" else min(ema50, ema200)
    entry_high   = max(min(price, ema20), min(price, ema50))
    stop_loss    = min(s20, price - atr_val * 1.5)
    target_1     = max(r20 * 1.02, price + (price - stop_loss) * 1.5)
    target_2     = max(r50 * 1.03, price + (price - stop_loss) * 3.0)
    risk_per_sh  = price - stop_loss
    reward_1     = target_1 - price
    rr_1         = reward_1 / risk_per_sh if risk_per_sh > 0 else 0

    ee_score   = 4 if trend_bias == "Bullish" else 3
    ee_details = [
        f"Entry Zone: **${entry_low:.2f} — ${entry_high:.2f}**",
        f"Stop Loss: **${stop_loss:.2f}** ({((price-stop_loss)/price*100):.1f}% risk from current price)",
        f"Target 1: **${target_1:.2f}** ({((target_1-price)/price*100):.1f}% upside) | R:R = **{rr_1:.1f}x**",
        f"Target 2: **${target_2:.2f}** ({((target_2-price)/price*100):.1f}% upside) | R:R = **{(target_2-price)/risk_per_sh:.1f}x**",
        f"Support 20-day: **${s20:.2f}** | Resistance 20-day: **${r20:.2f}**",
        f"Support 50-day: **${s50:.2f}** | Resistance 50-day: **${r50:.2f}**",
    ]
    ee_summary  = "\n".join(f"- {d}" for d in ee_details)
    ee_why_weak = "" if ee_score >= 3 else "R:R below 1.5x — not ideal entry conditions."

    # ── PILLAR 6: MINDSET ─────────────────────────────────────
    if vol >= 0.025:
        mind_score   = 3
        mind_summary = f"High volatility ({vol_pct:.2f}%/day) amplifies emotional mistakes. Rules: reduce size, honor your stop at ${stop_loss:.2f}, don't average down impulsively. ATR of ${atr_val:.2f} means normal daily swings will feel painful — plan for them."
    else:
        mind_score   = 4
        mind_summary = f"Stable trend reduces emotional pressure. Main risk is overconfidence — avoid chasing above ${r20:.2f} resistance. Always define your exit BEFORE entering."

    scores  = {"Fundamentals": fund_score, "Technicals": tech_score,
               "Risk Management": risk_score, "Trading Plan": plan_score,
               "Entry/Exit": ee_score, "Mindset": mind_score}
    overall = round(sum(scores.values()) / len(scores), 2)

    return {
        "price": price, "rsi": rsi_val, "trend_bias": trend_bias,
        "atr": atr_val, "atr_pct": atr_val / price * 100,
        "vol": vol_pct, "support_watch": s20, "resistance_watch": r20,
        "entry_low": entry_low, "entry_high": entry_high,
        "stop_loss": stop_loss, "target_1": target_1, "target_2": target_2,
        "rr": rr_1, "pos_size": pos_size,
        "fund_summary": fund_summary, "fund_why_weak": fund_why_weak,
        "tech_summary": tech_summary, "tech_why_weak": tech_why_weak,
        "risk_full": risk_full, "risk_why_weak": risk_why_weak,
        "plan_summary": plan_summary,
        "ee_summary": ee_summary, "ee_why_weak": ee_why_weak,
        "mind_summary": mind_summary,
        "scores": scores, "overall_score": overall,
        "levels": (s20, s50, s200, r20, r50, r200),
        "market_cap": market_cap, "sector": sector,
    }

# ─────────────────────────────────────────
# AI COMMENTARY
# ─────────────────────────────────────────
def ai_commentary(ticker, a, sent_label, sent_score):
    lines = []
    if a["overall_score"] >= 4:
        lines.append(f"**{ticker}** has a strong combined profile ({a['overall_score']:.1f}/5). Multiple pillars aligned.")
    elif a["overall_score"] >= 3:
        lines.append(f"**{ticker}** has a mixed setup ({a['overall_score']:.1f}/5). Trade with confirmation, not conviction.")
    else:
        lines.append(f"**{ticker}** has a weak combined setup ({a['overall_score']:.1f}/5). Capital preservation > aggressive positioning.")

    if a["trend_bias"] == "Bullish":
        lines.append(f"Trend is **BULLISH**. Buy pullbacks to ${a['entry_low']:.2f}–${a['entry_high']:.2f}. Don't chase extended candles.")
    elif a["trend_bias"] == "Bearish":
        lines.append(f"Trend is **BEARISH**. Avoid longs unless price reclaims EMA50. Wait for structure shift.")
    else:
        lines.append("Trend is **NEUTRAL**. Wait for clearer directional confirmation.")

    lines.append(f"Sentiment: **{sent_label}** ({sent_score}/100). Use as context, not your main signal.")
    lines.append(f"Support: **${a['support_watch']:.2f}** | Resistance: **${a['resistance_watch']:.2f}**")
    lines.append(f"Entry: **${a['entry_low']:.2f}–${a['entry_high']:.2f}** | Stop: **${a['stop_loss']:.2f}** | T1: **${a['target_1']:.2f}** | T2: **${a['target_2']:.2f}** | R:R: **{a['rr']:.1f}x**")
    return lines

# ─────────────────────────────────────────
# CHART — TradingView-style (unchanged)
# ─────────────────────────────────────────
def build_chart(df, ticker, show_bb=True, show_vwap=True, show_st=True, show_ema=True, show_macd=False, range_days=None):
    if range_days:
        df = df.tail(range_days)

    vol_colors = np.where(df["Close"] >= df["Open"], "#26a69a", "#ef5350")
    rows    = 3 if show_macd else 2
    heights = [0.60, 0.15, 0.25] if show_macd else [0.72, 0.28]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.01, row_heights=heights, subplot_titles=None)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price",
        increasing=dict(line=dict(color="#26a69a", width=1), fillcolor="#26a69a"),
        decreasing=dict(line=dict(color="#ef5350", width=1), fillcolor="#ef5350"),
        whiskerwidth=0,
    ), row=1, col=1)

    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper",
            line=dict(color="#5c6bc0", width=0.8, dash="dot"), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower",
            line=dict(color="#5c6bc0", width=0.8, dash="dot"),
            fill="tonexty", fillcolor="rgba(92,107,192,0.05)", showlegend=False), row=1, col=1)

    if show_ema:
        for col_name, color, lw, label in [
            ("EMA9",   "#ffffff", 1.0, "EMA 9"),
            ("EMA20",  "#f5c542", 1.0, "EMA 20"),
            ("EMA50",  "#4da3ff", 1.2, "EMA 50"),
            ("EMA200", "#b388ff", 1.2, "EMA 200"),
        ]:
            fig.add_trace(go.Scatter(x=df.index, y=df[col_name], mode="lines",
                name=label, line=dict(color=color, width=lw)), row=1, col=1)

    if show_vwap:
        fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], mode="lines",
            name="VWAP", line=dict(color="#ff9800", width=1.2, dash="dash")), row=1, col=1)

    if show_st:
        st_up   = df["Supertrend"].where(df["SupertrendDir"] == "up")
        st_down = df["Supertrend"].where(df["SupertrendDir"] == "down")
        fig.add_trace(go.Scatter(x=df.index, y=st_up, mode="lines",
            name="ST Bull", line=dict(color="#26a69a", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=st_down, mode="lines",
            name="ST Bear", line=dict(color="#ef5350", width=1.5)), row=1, col=1)

    s20, s50, s200, r20, r50, r200 = get_levels(df)
    for level, color, label in [(s20,"#26a69a","S20"),(r20,"#ef5350","R20")]:
        fig.add_hline(y=level, line_dash="dot", line_color=color, line_width=0.8,
                      annotation_text=f" {label}: {level:.2f}",
                      annotation_font_color=color, annotation_font_size=10, row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
        marker_color=vol_colors, showlegend=False), row=2, col=1)

    if show_macd:
        macd_colors = np.where(df["MACD_Hist"] >= 0, "#26a69a", "#ef5350")
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="MACD Hist",
            marker_color=macd_colors, showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines",
            name="MACD", line=dict(color="#2962ff", width=1.2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], mode="lines",
            name="Signal", line=dict(color="#ff6b35", width=1.0)), row=3, col=1)

    fig.update_layout(
        title=dict(text=f"  {ticker}", font=dict(size=16, color="#d1d4dc"), x=0),
        template="plotly_dark",
        height=720 if show_macd else 600,
        dragmode="pan",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=10, t=36, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=11)),
        plot_bgcolor="#131722",
        paper_bgcolor="#131722",
        font=dict(color="#d1d4dc", size=11),
        modebar=dict(bgcolor="#1e222d", color="#787b86", activecolor="#2962ff"),
        newshape=dict(line_color="#2962ff"),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="#1f2937", gridwidth=0.5,
        zeroline=False, showspikes=True, spikemode="across",
        spikesnap="cursor", spikecolor="#787b86", spikethickness=1,
        showline=True, linecolor="#363a45",
        rangeselector=dict(
            buttons=[
                dict(count=5,  label="5D",  step="day",   stepmode="backward"),
                dict(count=1,  label="1M",  step="month", stepmode="backward"),
                dict(count=3,  label="3M",  step="month", stepmode="backward"),
                dict(count=6,  label="6M",  step="month", stepmode="backward"),
                dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor="#1e222d", activecolor="#2962ff",
            font=dict(color="#d1d4dc", size=11),
            bordercolor="#363a45", borderwidth=1,
        ) if len(df) > 20 else None,
    )
    fig.update_yaxes(showgrid=True, gridcolor="#1f2937", gridwidth=0.5,
                     zeroline=False, showline=True, linecolor="#363a45",
                     tickformat=".2f", side="right")
    fig.update_yaxes(title_text="", row=2, col=1, tickformat=".2s")
    if show_macd:
        fig.update_yaxes(title_text="", row=3, col=1)
    return fig

# ─────────────────────────────────────────
# SCORE BADGE
# ─────────────────────────────────────────
def score_badge(score):
    if score >= 4: return "strong", "🟢 Strong"
    if score >= 3: return "avg",    "🟡 Average"
    return "weak", "🔴 Weak"

# ─────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Trading Desk")
    page = st.radio("Navigate", [
        "📈 Chart & Analysis",
        "💼 Portfolio",
        "👁️ Watchlist",
        "📓 Journal",
    ], label_visibility="collapsed")
    st.divider()
    st.markdown("**Quick Access**")
    quick_tickers = ["AVGO", "META", "NVDA", "AMD", "AMZN"]
    for qt in quick_tickers:
        if st.button(qt, key=f"quick_{qt}", use_container_width=True):
            st.session_state["ticker_input"] = qt
    st.divider()
    st.caption("Data: Yahoo Finance · Refresh every 5 min")

# ═══════════════════════════════════════════════════════════════
# PAGE 1: CHART & ANALYSIS
# ═══════════════════════════════════════════════════════════════
if "📈" in page:
    st.markdown("## 📈 Chart & Analysis")

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        default_tick = st.session_state.get("ticker_input", "AVGO")
        ticker = st.text_input("Ticker", value=default_tick, placeholder="e.g. AVGO, META, NVDA").strip().upper()
    with c2:
        interval = st.selectbox("Interval", ["1m","5m","15m","1h","4h","1d","1wk"], index=5)
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("▶ Load", use_container_width=True)

    if not ticker:
        st.info("Enter a ticker symbol above.")
        st.stop()

    df, info, error = get_stock_data(ticker, interval)
    if error or df is None:
        st.error(f"Could not load data for **{ticker}**: {error}")
        st.stop()

    news         = get_news(ticker)
    sent_score, sent_label = calculate_sentiment(news)
    a            = analyze(df, info or {}, ticker, sent_score)
    ai           = ai_commentary(ticker, a, sent_label, sent_score)
    catalyst     = get_catalyst_data(ticker)
    opt_data     = get_options_data(ticker)
    mtf          = get_mtf_data(ticker)

    latest_close = float(df["Close"].iloc[-1])
    prev_close   = float(df["Close"].iloc[-2]) if len(df) > 1 else latest_close
    chg          = latest_close - prev_close
    chg_pct      = chg / prev_close * 100 if prev_close else 0

    # ── Header Metrics ────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Ticker",    ticker)
    m2.metric("Price",     f"${latest_close:.2f}", f"{chg:+.2f} ({chg_pct:+.2f}%)")
    m3.metric("RSI",       f"{a['rsi']:.1f}", "Overbought" if a['rsi'] > 70 else ("Oversold" if a['rsi'] < 30 else "Normal"))
    m4.metric("Trend",     a["trend_bias"])
    m5.metric("Score",     f"{a['overall_score']}/5")
    m6.metric("Sentiment", f"{sent_score}/100 {sent_label}")

    st.divider()

    # ── CATALYST PANEL ────────────────────────────────────────
    st.markdown("### 🗓️ Catalyst Panel")
    st.markdown(f'<div class="{catalyst["risk_css"]}"><strong>{catalyst["risk_label"]}</strong></div>', unsafe_allow_html=True)
    st.markdown("")

    ca1, ca2, ca3, ca4 = st.columns(4)
    with ca1:
        if catalyst["earnings_date"]:
            st.metric("Next Earnings", str(catalyst["earnings_date"]),
                      f"{catalyst['days_to_earnings']} days" if catalyst["days_to_earnings"] is not None else "")
        else:
            st.metric("Next Earnings", "N/A")
    with ca2:
        st.metric("Event Risk", catalyst["event_risk"])
    with ca3:
        if catalyst["ex_div_date"]:
            st.metric("Ex-Div Date", str(catalyst["ex_div_date"]))
        else:
            st.metric("Ex-Div Date", "N/A")
    with ca4:
        if catalyst["analyst_target"]:
            upside = (catalyst["analyst_target"] - latest_close) / latest_close * 100
            st.metric("Analyst Target", f"${catalyst['analyst_target']:.2f}", f"{upside:+.1f}%")
        else:
            st.metric("Analyst Target", "N/A")

    st.divider()

    # ── Chart Controls ────────────────────────────────────────
    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
    with cc1: show_ema  = st.checkbox("EMAs",       value=True)
    with cc2: show_bb   = st.checkbox("Bollinger",  value=False)
    with cc3: show_vwap = st.checkbox("VWAP",       value=True)
    with cc4: show_st   = st.checkbox("Supertrend", value=True)
    with cc5: show_macd = st.checkbox("MACD Panel", value=False)

    st.caption("💡 Scroll to zoom · Drag to pan · 1M/3M/1Y buttons top-left · Double-click to reset")

    fig = build_chart(df, ticker, show_bb, show_vwap, show_st, show_ema, show_macd)
    st.plotly_chart(fig, use_container_width=True, config={
        "scrollZoom": True, "displayModeBar": True,
        "modeBarButtonsToAdd": ["drawline","drawopenpath","drawrect","eraseshape"],
        "modeBarButtonsToRemove": ["autoScale2d"],
        "toImageButtonOptions": {"format": "png", "filename": f"{ticker}_chart"},
    })

    st.divider()

    # ── MULTI-TIMEFRAME CONFIRMATION ──────────────────────────
    st.markdown("### 🕐 Multi-Timeframe Confirmation")

    align = mtf["alignment_score"]
    align_color = "#26a69a" if align >= 65 else ("#f5c842" if align >= 45 else "#ef5350")
    st.markdown(f"**Alignment Score:** <span style='color:{align_color};font-size:22px;font-weight:bold'>{align}/100</span>", unsafe_allow_html=True)

    mta, mtb, mtc, mtd = st.columns(4)
    mta.metric("Primary Bias",    mtf["primary_bias"])
    mtb.metric("Confirmation",    mtf["confirmation"][:35] + "…" if len(mtf["confirmation"]) > 35 else mtf["confirmation"])
    mtc.metric("Execution Trigger", mtf["trigger"][:35] + "…" if len(mtf["trigger"]) > 35 else mtf["trigger"])
    mtd.metric("Conflict Note",   mtf["conflict_note"][:35] + "…" if len(mtf["conflict_note"]) > 35 else mtf["conflict_note"])

    with st.expander("📋 Full Multi-Timeframe Table", expanded=False):
        mtf_df = pd.DataFrame(mtf["rows"])
        st.dataframe(mtf_df, use_container_width=True, hide_index=True)
        st.caption(f"Primary Bias: **{mtf['primary_bias']}** · Trigger: {mtf['trigger']} · {mtf['conflict_note']}")

    st.divider()

    # ── OPTIONS SETUP ─────────────────────────────────────────
    st.markdown("### 🎯 Options Setup")

    opt_vehicle, opt_reason = options_recommendation(
        a["trend_bias"], catalyst["event_risk"],
        opt_data.get("iv"), a["rsi"], a["atr_pct"]
    )

    oa, ob, oc, od = st.columns(4)
    oa.metric("Recommendation", opt_vehicle)
    oc.metric("IV (Median)",    f"{opt_data['iv']:.1f}%" if opt_data.get("iv") else "N/A")
    ob.metric("Next Expiry",    opt_data.get("next_exp") or "N/A")
    od.metric("Event Risk",     catalyst["event_risk"])

    st.info(f"**Reasoning:** {opt_reason}")

    if opt_data.get("error"):
        st.caption(f"⚠️ Options data note: {opt_data['error']}")

    st.divider()

    # ── AI Comments + News ────────────────────────────────────
    col_ai, col_news = st.columns([1, 1])

    with col_ai:
        st.markdown("### 🤖 AI Summary")
        sent_color = "#26a69a" if sent_label == "Bullish" else ("#ef5350" if sent_label == "Bearish" else "#f5c842")
        st.progress(sent_score / 100)
        st.markdown(f"<span style='color:{sent_color};font-weight:bold'>Sentiment: {sent_label} ({sent_score}/100)</span>", unsafe_allow_html=True)
        st.markdown("")
        for line in ai:
            st.markdown(f"▸ {line}")

    with col_news:
        st.markdown("### 📰 Latest News")
        for item in news[:6]:
            with st.container(border=True):
                st.markdown(f"**{item['title']}**")
                if item["published"]:
                    st.caption(item["published"])
                st.markdown(f"[Read →]({item['link']})")

    st.divider()

    # ── RISK CALCULATOR ───────────────────────────────────────
    st.markdown("### 💰 Risk Calculator")
    with st.expander("Open Position Sizer", expanded=False):
        rca, rcb = st.columns(2)
        with rca:
            rc_account  = st.number_input("Account Size ($)", value=50000.0, step=1000.0, key="rc_acct")
            rc_cash     = st.number_input("Cash Available ($)", value=16147.0, step=100.0, key="rc_cash")
            rc_risk_pct = st.number_input("Risk Per Trade (%)", value=1.0, min_value=0.1, max_value=10.0, step=0.1, key="rc_riskpct")
            rc_asset    = st.selectbox("Asset Type", ["Stock", "Option"], key="rc_asset")
        with rcb:
            rc_entry  = st.number_input("Entry Price ($)", value=float(f"{a['entry_high']:.2f}"), key="rc_entry")
            rc_stop   = st.number_input("Stop Loss ($)",  value=float(f"{a['stop_loss']:.2f}"), key="rc_stop")
            rc_t1     = st.number_input("Target 1 ($)",   value=float(f"{a['target_1']:.2f}"), key="rc_t1")
            rc_t2     = st.number_input("Target 2 ($)",   value=float(f"{a['target_2']:.2f}"), key="rc_t2")

        if st.button("Calculate Position Size", key="calc_pos"):
            result, warning = calc_position_size(
                rc_account, rc_cash, rc_risk_pct,
                rc_entry, rc_stop, rc_t1, rc_t2,
                rc_asset, a["atr"]
            )
            if result is None:
                st.error(warning)
            else:
                p1, p2, p3, p4, p5, p6 = st.columns(6)
                p1.metric("Max $ Risk",    f"${result['max_risk_dollars']:,.2f}")
                p2.metric(f"{rc_asset}s",  str(result["units"]) + " " + result["label_unit"])
                p3.metric("Position Value",f"${result['pos_value']:,.2f}")
                p4.metric("Risk/Unit",     f"${result['risk_per_unit']:.2f}")
                p5.metric("R:R → T1",      f"{result['rr1']:.2f}x")
                p6.metric("R:R → T2",      f"{result['rr2']:.2f}x")
                if warning:
                    st.warning(warning)
                st.session_state["last_position_calc"] = {
                    "ticker": ticker, "entry": rc_entry, "stop": rc_stop,
                    "target1": rc_t1, "target2": rc_t2, "size": result["units"],
                    "asset_type": rc_asset, "rr1": result["rr1"],
                }

    st.divider()

    # ── 6 PILLARS ─────────────────────────────────────────────
    st.markdown("### 📐 6-Pillar Analysis")

    pillar_data = [
        ("1. Fundamentals",     a["scores"]["Fundamentals"],    a["fund_summary"],  a["fund_why_weak"]),
        ("2. Technicals",       a["scores"]["Technicals"],       a["tech_summary"],  a["tech_why_weak"]),
        ("3. Risk Management",  a["scores"]["Risk Management"],  a["risk_full"],     a["risk_why_weak"]),
        ("4. Trading Plan",     a["scores"]["Trading Plan"],     a["plan_summary"],  ""),
        ("5. Entry / Exit",     a["scores"]["Entry/Exit"],       a["ee_summary"],    a["ee_why_weak"]),
        ("6. Mindset",          a["scores"]["Mindset"],          a["mind_summary"],  ""),
    ]

    for title, score, summary, why_weak in pillar_data:
        css_class, badge = score_badge(score)
        with st.expander(f"{badge}  {title}  —  {score}/5", expanded=(score <= 2)):
            st.markdown(summary)
            if why_weak:
                st.error(why_weak)

    st.divider()

    st.markdown("### 📊 Summary Table")
    table_data = pd.DataFrame({
        "Pillar": list(a["scores"].keys()),
        "Score":  list(a["scores"].values()),
        "Rating": [score_badge(v)[1] for v in a["scores"].values()],
    })
    st.dataframe(table_data, use_container_width=True, hide_index=True)

    st.success(
        f"**Overall:** {a['overall_score']}/5  |  "
        f"**Trend:** {a['trend_bias']}  |  "
        f"**Entry:** ${a['entry_low']:.2f}–${a['entry_high']:.2f}  |  "
        f"**Stop:** ${a['stop_loss']:.2f}  |  "
        f"**T1:** ${a['target_1']:.2f}  |  "
        f"**T2:** ${a['target_2']:.2f}  |  "
        f"**R:R:** {a['rr']:.1f}x"
    )

    st.divider()

    # ── SAVE TO JOURNAL BUTTON ────────────────────────────────
    st.markdown("### 💾 Save Setup to Journal")
    with st.expander("Save this setup as a trade entry", expanded=False):
        j1, j2 = st.columns(2)
        with j1:
            j_direction  = st.selectbox("Direction", ["Long", "Short"], key="j_dir")
            j_asset_type = st.selectbox("Asset Type", ["Stock", "Option", "Spread"], key="j_at")
            j_thesis     = st.text_area("Trade Thesis", placeholder="Why are you taking this trade?", key="j_thesis")
            j_timeframe  = st.selectbox("Primary Timeframe", ["1m","5m","15m","1h","4h","1d"], index=5, key="j_tf")
        with j2:
            j_result  = st.selectbox("Result (if closing)", ["Open","Win","Loss","Breakeven"], key="j_res")
            j_pnl     = st.number_input("P&L ($)", value=0.0, key="j_pnl")
            j_notes   = st.text_area("Notes", placeholder="What happened?", key="j_notes")
            j_mistakes= st.text_area("Mistakes / Lessons", placeholder="What would you do differently?", key="j_mistakes")

        lpc = st.session_state.get("last_position_calc", {})

        if st.button("💾 Save to Journal", key="save_journal"):
            trade_row = {
                "date":            datetime.now().strftime("%Y-%m-%d %H:%M"),
                "ticker":          ticker,
                "direction":       j_direction,
                "asset_type":      j_asset_type,
                "entry":           lpc.get("entry", a["entry_high"]),
                "stop":            lpc.get("stop",  a["stop_loss"]),
                "target":          lpc.get("target1", a["target_1"]),
                "size":            lpc.get("size", ""),
                "thesis":          j_thesis,
                "timeframe":       j_timeframe,
                "alignment_score": mtf["alignment_score"],
                "catalyst_state":  catalyst["risk_label"],
                "options_setup":   opt_vehicle,
                "result":          j_result,
                "pnl":             j_pnl,
                "notes":           j_notes,
                "mistakes":        j_mistakes,
            }
            save_trade(trade_row)
            st.success(f"✅ Trade saved to journal: {ticker} {j_direction} on {trade_row['date']}")

# ═══════════════════════════════════════════════════════════════
# PAGE 2: PORTFOLIO
# ═══════════════════════════════════════════════════════════════
elif "💼" in page:
    st.markdown("## 💼 Portfolio")

    # Initialize portfolio in session state (persists until page refresh)
    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = DEFAULT_PORTFOLIO.copy()

    portfolio = st.session_state["portfolio"]

    # Add / Update position
    with st.expander("➕ Add / Update Position"):
        pa, pb, pc, pd_col, pe = st.columns(5)
        with pa:   new_ticker = st.text_input("Ticker (exact, e.g. AVGO.NE)", key="new_tick").strip().upper()
        with pb:   new_shares = st.number_input("Shares", min_value=0.0, key="new_shares")
        with pc:   new_avg    = st.number_input("Avg Cost", min_value=0.0, key="new_avg")
        with pd_col: new_cur  = st.selectbox("Currency", ["CAD", "USD"], key="new_cur")
        with pe:   new_target = st.number_input("Target", min_value=0.0, key="new_target")
        if st.button("Save Position") and new_ticker:
            st.session_state["portfolio"][new_ticker] = {
                "shares": new_shares, "avg_cost": new_avg,
                "currency": new_cur, "target": new_target
            }
            st.success(f"Saved {new_ticker}")
            st.rerun()

    # Remove position
    if portfolio:
        remove_tick = st.selectbox("Remove Position", ["—"] + list(portfolio.keys()))
        if st.button("Remove") and remove_tick != "—":
            del st.session_state["portfolio"][remove_tick]
            st.success(f"Removed {remove_tick}")
            st.rerun()

    st.divider()
    st.markdown("### Current Positions")
    st.caption("Prices are fetched using the exact ticker you saved (e.g. AVGO.NE pulls from the NEO Exchange).")

    total_invested = 0
    total_value    = 0
    rows           = []

    for tick, pos in st.session_state["portfolio"].items():
        # Use exact ticker for price — no stripping of exchange suffix
        cur_price = get_portfolio_price(tick)
        if cur_price is None:
            cur_price = pos["avg_cost"]

        invested   = pos["shares"] * pos["avg_cost"]
        value      = pos["shares"] * cur_price
        pnl        = value - invested
        pnl_pct    = pnl / invested * 100 if invested else 0
        to_target  = ((pos["target"] - cur_price) / cur_price * 100) if pos["target"] and cur_price else 0

        total_invested += invested
        total_value    += value

        rows.append({
            "Ticker":    tick,
            "Shares":    pos["shares"],
            "Avg Cost":  f"${pos['avg_cost']:.4f}",
            "Cur Price": f"${cur_price:.2f}",
            "Value":     f"${value:,.2f}",
            "P&L":       f"${pnl:+,.2f} ({pnl_pct:+.2f}%)",
            "Target":    f"${pos['target']:.2f} (+{to_target:.1f}%)",
            "Currency":  pos["currency"],
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    total_pnl = total_value - total_invested
    st.divider()
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Total Invested", f"${total_invested:,.2f}")
    t2.metric("Total Value",    f"${total_value:,.2f}")
    t3.metric("Total P&L",      f"${total_pnl:+,.2f}", f"{total_pnl/total_invested*100:+.2f}%" if total_invested else "0%")
    t4.metric("Goal ($60k)",    f"{total_value/60000*100:.1f}%")

    st.divider()
    st.markdown("### Cash Reserve")
    ca, cb = st.columns(2)
    with ca: st.metric("USD Cash", "$16,147")
    with cb: st.metric("Estimated CAD (×1.38)", f"${16147*1.38:,.0f}")

# ═══════════════════════════════════════════════════════════════
# PAGE 3: WATCHLIST
# ═══════════════════════════════════════════════════════════════
elif "👁️" in page:
    st.markdown("## 👁️ Watchlist")

    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = DEFAULT_WATCHLIST.copy()

    wa, wb = st.columns([3, 1])
    with wa: add_tick = st.text_input("Add ticker to watchlist").strip().upper()
    with wb:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Add") and add_tick and add_tick not in st.session_state["watchlist"]:
            st.session_state["watchlist"].append(add_tick)
            st.success(f"Added {add_tick}")
            st.rerun()

    if st.session_state["watchlist"]:
        remove_w = st.selectbox("Remove from watchlist", ["—"] + st.session_state["watchlist"])
        if st.button("Remove from watchlist") and remove_w != "—":
            st.session_state["watchlist"].remove(remove_w)
            st.success(f"Removed {remove_w}")
            st.rerun()

    st.divider()
    st.markdown("### Live Scan")

    wl_rows = []
    for tick in st.session_state["watchlist"]:
        df_w, info_w, _ = get_stock_data(tick, "1d")
        if df_w is None or df_w.empty:
            continue
        latest_w = df_w.iloc[-1]
        prev_w   = df_w.iloc[-2] if len(df_w) > 1 else latest_w
        price_w  = float(latest_w["Close"])
        chg_w    = (price_w - float(prev_w["Close"])) / float(prev_w["Close"]) * 100
        rsi_w    = float(latest_w["RSI"]) if "RSI" in df_w.columns else 0
        st_dir_w = str(latest_w["SupertrendDir"]) if "SupertrendDir" in df_w.columns else "—"
        trend_w  = "▲ Bull" if price_w > float(latest_w.get("EMA50", price_w)) else "▼ Bear"
        signal_w = "🟢 BUY" if (st_dir_w == "up" and rsi_w < 65) else ("🔴 SELL" if st_dir_w == "down" else "🟡 WAIT")

        wl_rows.append({
            "Ticker": tick,
            "Price":  f"${price_w:.2f}",
            "Chg%":   f"{chg_w:+.2f}%",
            "RSI":    f"{rsi_w:.1f}",
            "Trend":  trend_w,
            "ST Dir": st_dir_w.upper(),
            "Signal": signal_w,
        })

    if wl_rows:
        st.dataframe(pd.DataFrame(wl_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Loading watchlist data...")

# ═══════════════════════════════════════════════════════════════
# PAGE 4: JOURNAL & PERFORMANCE
# ═══════════════════════════════════════════════════════════════
elif "📓" in page:
    st.markdown("## 📓 Trade Journal & Performance")

    journal_df = load_journal()
    analytics  = journal_analytics(journal_df)

    # ── Analytics Summary ─────────────────────────────────────
    if analytics:
        st.markdown("### 📊 Performance Analytics")
        ja1, ja2, ja3, ja4 = st.columns(4)
        ja1.metric("Total Trades",   analytics["total"])
        ja2.metric("Win Rate",       f"{analytics['win_rate']:.1f}%")
        ja3.metric("Total P&L",      f"${analytics['total_pnl']:+,.2f}")
        ja4.metric("Profit Factor",  f"{analytics['profit_factor']:.2f}x" if analytics['profit_factor'] != float('inf') else "∞")

        jb1, jb2, jb3, _ = st.columns(4)
        jb1.metric("Avg Win",        f"${analytics['avg_win']:+,.2f}")
        jb2.metric("Avg Loss",       f"${analytics['avg_loss']:+,.2f}")
        jb3.metric("Expectancy",     f"${analytics['expectancy']:+,.2f}")

        if analytics["insights"]:
            st.markdown("#### 🔍 Automatic Insights")
            for insight in analytics["insights"]:
                st.info(insight)

        # P&L chart
        if not journal_df.empty and "pnl" in journal_df.columns:
            pnl_series = pd.to_numeric(journal_df["pnl"], errors="coerce").fillna(0)
            if pnl_series.sum() != 0:
                fig_pnl = go.Figure()
                fig_pnl.add_trace(go.Bar(
                    x=list(range(1, len(pnl_series)+1)),
                    y=pnl_series.tolist(),
                    marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in pnl_series],
                    name="P&L per Trade"
                ))
                fig_pnl.update_layout(
                    template="plotly_dark", plot_bgcolor="#131722", paper_bgcolor="#131722",
                    height=250, margin=dict(l=10,r=10,t=30,b=10),
                    title="P&L per Trade",
                    font=dict(color="#d1d4dc"),
                )
                st.plotly_chart(fig_pnl, use_container_width=True)

        st.divider()

    # ── Add New Trade ─────────────────────────────────────────
    st.markdown("### ➕ Log New Trade")
    with st.expander("Open Trade Entry Form", expanded=not analytics):
        f1, f2 = st.columns(2)
        with f1:
            f_ticker    = st.text_input("Ticker", key="f_tick")
            f_direction = st.selectbox("Direction", ["Long","Short"], key="f_dir")
            f_asset     = st.selectbox("Asset Type", ["Stock","Option","Spread"], key="f_ast")
            f_entry     = st.number_input("Entry ($)", value=0.0, key="f_ent")
            f_stop      = st.number_input("Stop ($)",  value=0.0, key="f_stp")
            f_target    = st.number_input("Target ($)", value=0.0, key="f_tgt")
            f_size      = st.number_input("Size (shares/contracts)", value=0, key="f_sz")
        with f2:
            f_thesis    = st.text_area("Thesis", key="f_th")
            f_tf        = st.selectbox("Timeframe", ["1m","5m","15m","1h","4h","1d"], index=5, key="f_tff")
            f_align     = st.slider("Alignment Score (0–100)", 0, 100, 50, key="f_aln")
            f_catalyst  = st.selectbox("Catalyst State", ["Low Risk","Moderate Risk","High Risk"], key="f_cat")
            f_opt_setup = st.text_input("Options Setup Used", key="f_opt")
            f_result    = st.selectbox("Result", ["Open","Win","Loss","Breakeven"], key="f_res")
            f_pnl       = st.number_input("P&L ($)", value=0.0, key="f_pnl")
            f_notes     = st.text_area("Notes", key="f_nt")
            f_mistakes  = st.text_area("Mistakes / Lessons", key="f_mis")

        if st.button("Save Trade", key="save_manual_trade"):
            if f_ticker:
                save_trade({
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "ticker": f_ticker.upper(), "direction": f_direction,
                    "asset_type": f_asset, "entry": f_entry, "stop": f_stop,
                    "target": f_target, "size": f_size, "thesis": f_thesis,
                    "timeframe": f_tf, "alignment_score": f_align,
                    "catalyst_state": f_catalyst, "options_setup": f_opt_setup,
                    "result": f_result, "pnl": f_pnl, "notes": f_notes,
                    "mistakes": f_mistakes,
                })
                st.success("Trade logged!")
                st.rerun()
            else:
                st.error("Please enter a ticker.")

    st.divider()

    # ── Trade Log Table ───────────────────────────────────────
    st.markdown("### 📋 Trade Log")
    if journal_df.empty:
        st.info("No trades logged yet. Use the form above or the 'Save to Journal' button on the Chart page.")
    else:
        st.dataframe(journal_df, use_container_width=True, hide_index=True)
        # Download
        csv_str = journal_df.to_csv(index=False)
        st.download_button("⬇ Download Journal CSV", data=csv_str,
                           file_name="trade_journal.csv", mime="text/csv")

        # Delete last trade
        if st.button("🗑 Delete Last Trade", key="del_last"):
            if not journal_df.empty:
                journal_df = journal_df.iloc[:-1]
                journal_df.to_csv(JOURNAL_FILE, index=False)
                st.success("Last trade deleted.")
                st.rerun()
