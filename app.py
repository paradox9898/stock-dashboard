import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os, csv, json, requests, re

st.set_page_config(page_title="Utpal Trading Desk", layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════
#  CSS — clean TradingView-style dark with card polish
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
body,.stApp{background:#131722;color:#d1d4dc;font-family:'Inter',sans-serif;}
.stTextInput>div>input,.stSelectbox>div>div,.stNumberInput>div>input{
  background:#1e222d;color:#d1d4dc;border:1px solid #363a45;border-radius:6px;}
.stButton>button{background:#2962ff;color:#fff;border:none;border-radius:6px;
  padding:7px 18px;font-weight:600;letter-spacing:.3px;}
.stButton>button:hover{background:#1e53e5;}
/* Cards */
.card{background:#1e222d;border:1px solid #2a2e3e;border-radius:10px;padding:16px 20px;margin-bottom:12px;}
.card-accent-green{border-left:4px solid #26a69a;}
.card-accent-red  {border-left:4px solid #ef5350;}
.card-accent-blue {border-left:4px solid #2962ff;}
.card-accent-gold {border-left:4px solid #f5c842;}
/* Metric overrides */
div[data-testid="stMetric"]{background:#1e222d;border:1px solid #2a2e3e;
  border-radius:8px;padding:10px 14px;}
div[data-testid="stMetricLabel"]{color:#787b86;font-size:11px;text-transform:uppercase;letter-spacing:.5px;}
div[data-testid="stMetricValue"]{color:#d1d4dc;font-size:20px;font-weight:700;}
/* Sidebar */
[data-testid="stSidebar"]{background:#161b27;border-right:1px solid #2a2e3e;}
/* Risk badges */
.risk-high{background:rgba(239,83,80,.15);border:1px solid #ef5350;border-radius:8px;padding:10px 16px;}
.risk-med {background:rgba(245,200,66,.10);border:1px solid #f5c842;border-radius:8px;padding:10px 16px;}
.risk-low {background:rgba(38,166,154,.10);border:1px solid #26a69a;border-radius:8px;padding:10px 16px;}
/* Signal chips */
.chip{display:inline-block;padding:4px 10px;border-radius:20px;font-size:11px;
  font-weight:600;margin:3px 3px 3px 0;white-space:nowrap;}
.chip-green{background:rgba(38,166,154,.2);border:1px solid #26a69a;color:#26a69a;}
.chip-red  {background:rgba(239,83,80,.2);border:1px solid #ef5350;color:#ef5350;}
.chip-gold {background:rgba(245,200,66,.2);border:1px solid #f5c842;color:#f5c842;}
.chip-blue {background:rgba(41,98,255,.2);border:1px solid #2962ff;color:#7ab4ff;}
/* AI box */
.ai-box{background:#161b27;border:1px solid #2962ff;border-radius:10px;
  padding:18px 22px;margin-top:10px;line-height:1.7;}
/* News scroll */
.news-scroll{max-height:420px;overflow-y:auto;padding-right:6px;}
.news-item{background:#1e222d;border:1px solid #2a2e3e;border-radius:8px;
  padding:10px 14px;margin-bottom:8px;}
.news-item:hover{border-color:#2962ff;}
/* Trade plan table */
.trade-table{width:100%;border-collapse:collapse;}
.trade-table td{padding:6px 4px;border-bottom:1px solid #2a2e3e;font-size:13px;}
.trade-table td:first-child{color:#787b86;}
.trade-table td:last-child{text-align:right;font-weight:600;}
/* Signal alert */
.signal-alert{background:rgba(239,83,80,.12);border:1px solid #ef5350;
  border-radius:8px;padding:10px 14px;margin-bottom:8px;}
.signal-good {background:rgba(38,166,154,.12);border:1px solid #26a69a;
  border-radius:8px;padding:10px 14px;margin-bottom:8px;}
/* Data integrity */
.data-panel{background:#161b27;border:1px solid #2a2e3e;border-radius:8px;padding:12px 16px;margin-bottom:12px;}
.green{color:#26a69a;font-weight:700;}
.red{color:#ef5350;font-weight:700;}
.gold{color:#f5c842;font-weight:700;}
h1,h2,h3{color:#d1d4dc;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PORTFOLIO CONFIG
# ══════════════════════════════════════════════════════════════
DEFAULT_PORTFOLIO = {
    "AVGO.TO": {"shares": 1005, "avg_cost": 13.4353, "currency": "CAD", "target": 20.0},
    "META.TO": {"shares": 417,  "avg_cost": 32.59,   "currency": "CAD", "target": 45.0},
}
CASH_USD   = 16147.0
CAD_RATE   = 1.38
GOAL_LOW   = 60000
GOAL_HIGH  = 80000

DEFAULT_WATCHLIST = [
    "AVGO","META","NVDA","AMD","AMZN","MSFT","GOOGL",
    "DOCN","NET","COHR","MU","INTC","NFLX","CLS",
    "ALAB","VRT","DELL","CIEN","GLW","KEYS",
]
JOURNAL_FILE = "trade_journal.csv"
JOURNAL_COLS = ["date","ticker","direction","asset_type","entry","stop","target",
                "size","thesis","timeframe","alignment_score","catalyst_state",
                "options_setup","result","pnl","notes","mistakes"]

# ══════════════════════════════════════════════════════════════
#  INDICATOR FUNCTIONS — all fixed/verified
# ══════════════════════════════════════════════════════════════
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.where(delta>0, 0).ewm(com=period-1, adjust=False).mean()
    loss  = (-delta.where(delta<0, 0)).ewm(com=period-1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    fast_ema   = series.ewm(span=fast, adjust=False).mean()
    slow_ema   = series.ewm(span=slow, adjust=False).mean()
    macd_line  = fast_ema - slow_ema
    signal_line= macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def atr(df, period=14):
    hl  = df["High"] - df["Low"]
    hc  = (df["High"] - df["Close"].shift(1)).abs()
    lc  = (df["Low"]  - df["Close"].shift(1)).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(com=period-1, adjust=False).mean()

def vwap_calc(df, interval="1d"):
    """VWAP: daily-reset for intraday, rolling-20 for daily/weekly."""
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    tpv = tp * df["Volume"]
    if interval in ("1m","5m","15m","1h","4h"):
        idx = df.index
        if hasattr(idx, "tz") and idx.tz is not None:
            dates = idx.tz_convert("America/New_York").normalize()
        else:
            dates = pd.DatetimeIndex(idx).normalize()
        result = pd.Series(np.nan, index=idx)
        for d in dates.unique():
            mask = dates == d
            cum_tpv = tpv[mask].cumsum()
            cum_vol = df["Volume"][mask].cumsum().replace(0, np.nan)
            result[mask] = cum_tpv.values / cum_vol.values
        return result
    else:
        # 20-period rolling VWAP for daily/weekly — volume-weighted MA
        roll_tpv = tpv.rolling(20, min_periods=1).sum()
        roll_vol = df["Volume"].rolling(20, min_periods=1).sum().replace(0, np.nan)
        return roll_tpv / roll_vol

def bollinger_bands(series, period=20, std_dev=2):
    mid   = series.rolling(period).mean()
    sigma = series.rolling(period).std()
    return mid + std_dev*sigma, mid, mid - std_dev*sigma

def supertrend(df, period=10, multiplier=3):
    data  = df.copy()
    hl2   = (data["High"] + data["Low"]) / 2
    data["ATR14"] = atr(data, period)
    ub = hl2 + multiplier * data["ATR14"]
    lb = hl2 - multiplier * data["ATR14"]
    final_ub, final_lb = ub.copy(), lb.copy()
    for i in range(1, len(data)):
        final_ub.iloc[i] = (min(ub.iloc[i], final_ub.iloc[i-1])
                            if data["Close"].iloc[i-1] <= final_ub.iloc[i-1] else ub.iloc[i])
        final_lb.iloc[i] = (max(lb.iloc[i], final_lb.iloc[i-1])
                            if data["Close"].iloc[i-1] >= final_lb.iloc[i-1] else lb.iloc[i])
    trend = pd.Series(index=data.index, dtype="float64")
    direction = pd.Series(index=data.index, dtype="object")
    for i in range(len(data)):
        if i == 0:
            trend.iloc[0] = final_lb.iloc[0]; direction.iloc[0] = "up"; continue
        if trend.iloc[i-1] == final_ub.iloc[i-1]:
            if data["Close"].iloc[i] <= final_ub.iloc[i]:
                trend.iloc[i] = final_ub.iloc[i]; direction.iloc[i] = "down"
            else:
                trend.iloc[i] = final_lb.iloc[i]; direction.iloc[i] = "up"
        else:
            if data["Close"].iloc[i] >= final_lb.iloc[i]:
                trend.iloc[i] = final_lb.iloc[i]; direction.iloc[i] = "up"
            else:
                trend.iloc[i] = final_ub.iloc[i]; direction.iloc[i] = "down"
    return trend, direction

# ── SUPPORT / RESISTANCE (swing highs/lows — not simple min/max) ──
def get_swing_levels(df, lookback=5):
    """Proper swing highs and lows via local extrema detection."""
    highs, lows = [], []
    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    n = len(df)
    for i in range(lookback, n - lookback):
        if h[i] == max(h[i-lookback:i+lookback+1]):
            highs.append(float(h[i]))
        if l[i] == min(l[i-lookback:i+lookback+1]):
            lows.append(float(l[i]))
    # Deduplicate levels within 1%
    def dedup(levels, reverse=False):
        seen, out = [], []
        for p in sorted(levels, reverse=reverse):
            if not any(abs(p - s)/max(s,0.01) < 0.01 for s in seen):
                seen.append(p); out.append(p)
        return out[:6]
    return dedup(highs, reverse=True), dedup(lows)

def get_key_levels(df):
    """Returns (s1,s2,s3,r1,r2,r3) using swing levels with fallback."""
    price = float(df["Close"].iloc[-1])
    swing_highs, swing_lows = get_swing_levels(df)
    supports    = sorted([x for x in swing_lows  if x < price*1.005], reverse=True)
    resistances = sorted([x for x in swing_highs if x > price*0.995])
    # Fallback to period min/max
    fb_s = [float(df["Low"].tail(n).min()) for n in [20,50,200] if len(df)>=n]
    fb_r = [float(df["High"].tail(n).max()) for n in [20,50,200] if len(df)>=n]
    while len(supports)    < 3: supports.append(fb_s[min(len(supports),len(fb_s)-1)] if fb_s else price*0.95)
    while len(resistances) < 3: resistances.append(fb_r[min(len(resistances),len(fb_r)-1)] if fb_r else price*1.05)
    return supports[0],supports[1],supports[2], resistances[0],resistances[1],resistances[2]

# ══════════════════════════════════════════════════════════════
#  DATA FETCH
# ══════════════════════════════════════════════════════════════
INTERVAL_MAP = {
    "1m":"7d","5m":"30d","15m":"60d","1h":"730d",
    "4h":"730d","1d":"2y","1wk":"5y",
}

@st.cache_data(ttl=300)
def get_stock_data(ticker, interval="1d"):
    try:
        period     = INTERVAL_MAP.get(interval, "2y")
        fetch_time = datetime.now()
        stock      = yf.Ticker(ticker)
        df         = stock.history(period=period, interval=interval, auto_adjust=False)
        if df is None or df.empty:
            return None, None, f"No data for {ticker}", None
        df = df.dropna().copy()
        # Core indicators
        df["EMA9"]   = ema(df["Close"],9)
        df["EMA20"]  = ema(df["Close"],20)
        df["EMA50"]  = ema(df["Close"],50)
        df["EMA200"] = ema(df["Close"],200)
        df["RSI"]    = rsi(df["Close"])
        df["ATR"]    = atr(df,14)
        df["VWAP"]   = vwap_calc(df, interval)
        df["MACD"],df["MACD_Signal"],df["MACD_Hist"] = macd(df["Close"])
        df["BB_Upper"],df["BB_Mid"],df["BB_Lower"]   = bollinger_bands(df["Close"])
        df["Supertrend"],df["SupertrendDir"]         = supertrend(df,10,3)
        # Volume indicators
        df["Vol_EMA20"] = ema(df["Volume"], 20)
        try:
            info = stock.info or {}
        except Exception:
            info = {}
        meta = {"fetch_time": fetch_time, "last_candle": df.index[-1],
                "bar_count": len(df), "interval": interval, "source": "Yahoo Finance"}
        return df, info, None, meta
    except Exception as e:
        return None, None, str(e), None

@st.cache_data(ttl=300)
def get_portfolio_price(ticker):
    try:
        df = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=False)
        if df is not None and not df.empty:
            prev = float(df["Close"].iloc[-2]) if len(df)>1 else float(df["Close"].iloc[-1])
            curr = float(df["Close"].iloc[-1])
            return curr, prev
    except Exception:
        pass
    return None, None

# ══════════════════════════════════════════════════════════════
#  MARKET REGIME
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=600)
def get_market_regime():
    result = {"SPY":"Unknown","QQQ":"Unknown","VIX":None,"regime":"Unknown","fear_greed":None}
    for sym in ["SPY","QQQ"]:
        try:
            df = yf.Ticker(sym).history(period="60d",interval="1d",auto_adjust=False)
            if df is not None and not df.empty:
                p  = float(df["Close"].iloc[-1])
                e20= float(df["Close"].ewm(span=20,adjust=False).mean().iloc[-1])
                e50= float(df["Close"].ewm(span=50,adjust=False).mean().iloc[-1])
                result[sym] = "Bullish" if (p>e20 and p>e50) else ("Bearish" if (p<e20 and p<e50) else "Chop")
        except Exception:
            pass
    try:
        vix = yf.Ticker("^VIX").history(period="5d",interval="1d",auto_adjust=False)
        if vix is not None and not vix.empty:
            result["VIX"] = float(vix["Close"].iloc[-1])
    except Exception:
        pass
    s,q,v = result["SPY"],result["QQQ"],result["VIX"]
    if s=="Bullish" and q=="Bullish":
        result["regime"] = "Risk-on" if not v or v<25 else "Cautious Bull"
    elif s=="Bearish" and q=="Bearish":
        result["regime"] = "Risk-off"
    elif v and v>35:
        result["regime"] = "Extreme Fear"
    else:
        result["regime"] = "Choppy"
    return result

# ══════════════════════════════════════════════════════════════
#  NEWS + SENTIMENT (multi-source, 20+ items)
# ══════════════════════════════════════════════════════════════
BULL_WORDS = {"beat","beats","surge","surges","strong","upgrade","upgrades","record","growth",
              "profit","soar","soars","buy","outperform","expands","positive","rebound","bullish",
              "momentum","raises","raised","tops","rally","rallies","breakout","accelerates",
              "boosts","exceed","exceeds","above","upside","outpaces","winning","strength"}
BEAR_WORDS = {"miss","misses","drop","drops","downgrade","downgrades","warning","weak","fall",
              "falls","lawsuit","loss","losses","decline","declines","probe","investigation",
              "negative","concern","concerns","bearish","cut","cuts","risks","below","selloff",
              "sell-off","layoffs","layoff","slows","slowdown","disappoints","disappointing",
              "warning","warns","withdraws","reduces","faces","headwinds"}

@st.cache_data(ttl=900)
def get_news(ticker):
    base  = ticker.split(".")[0]
    items = []
    queries = [
        f"{base}+stock",
        f"{base}+earnings+revenue",
        f"{base}+analyst+price+target",
        f"{base}+semiconductor" if base in ["AVGO","AMD","NVDA","MU","INTC","COHR"] else f"{base}+AI",
    ]
    seen_titles = set()
    for q in queries:
        try:
            feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en")
            for e in feed.entries[:8]:
                title = e.title[:120]
                if title not in seen_titles:
                    seen_titles.add(title)
                    items.append({"title": title, "link": e.link,
                                  "published": getattr(e,"published",""),
                                  "source": getattr(e,"source",{}).get("title","Google News") if hasattr(e,"source") else "Google News"})
        except Exception:
            pass
    # Yahoo Finance RSS
    try:
        yf_feed = feedparser.parse(f"https://finance.yahoo.com/rss/headline?s={base}")
        for e in yf_feed.entries[:6]:
            title = e.title[:120]
            if title not in seen_titles:
                seen_titles.add(title)
                items.append({"title": title, "link": e.link,
                              "published": getattr(e,"published",""), "source": "Yahoo Finance"})
    except Exception:
        pass
    return items[:25]

def calculate_sentiment(news_items):
    score, hits = 0, 0
    for item in news_items:
        t    = item["title"].lower()
        bull = sum(1 for w in BULL_WORDS if w in t)
        bear = sum(1 for w in BEAR_WORDS if w in t)
        weight = 1.5 if "analyst" in t or "target" in t or "upgrade" in t or "downgrade" in t else 1.0
        score += (bull - bear) * weight
        hits  += (bull + bear) * weight
    norm  = 50 if hits==0 else int(max(0, min(100, 50 + max(-42, min(42, score*7)))))
    label = "Bullish" if norm>=63 else ("Bearish" if norm<=37 else "Neutral")
    return norm, label

# ══════════════════════════════════════════════════════════════
#  CATALYST / EARNINGS
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def get_catalyst_data(ticker):
    result = {"earnings_date":None,"days_to_earnings":None,"ex_div_date":None,
              "analyst_target":None,"event_risk":"Low","risk_label":"🟢 Low Risk","risk_css":"risk-low"}
    try:
        stock = yf.Ticker(ticker)
        cal   = stock.calendar
        info  = stock.info or {}
        if cal is not None:
            try:
                ed_raw = None
                if hasattr(cal,"columns") and "Earnings Date" in cal.columns:
                    ed_raw = cal["Earnings Date"].iloc[0]
                elif isinstance(cal, dict) and "Earnings Date" in cal:
                    ed_raw = cal["Earnings Date"]
                    if isinstance(ed_raw, list): ed_raw = ed_raw[0]
                if ed_raw is not None:
                    ed  = pd.Timestamp(ed_raw).date() if not hasattr(ed_raw,"date") else ed_raw.date()
                    dte = (ed - datetime.now().date()).days
                    result.update({"earnings_date":ed,"days_to_earnings":dte})
                    if dte<=5:
                        result.update({"event_risk":"High","risk_label":"🔴 High Risk — Earnings ≤5d","risk_css":"risk-high"})
                    elif dte<=14:
                        result.update({"event_risk":"Moderate","risk_label":"🟡 Moderate — Earnings ≤14d","risk_css":"risk-med"})
            except Exception:
                pass
        try:
            ex = info.get("exDividendDate")
            if ex: result["ex_div_date"] = pd.Timestamp(ex,unit="s").date()
        except Exception:
            pass
        result["analyst_target"] = info.get("targetMeanPrice")
    except Exception:
        pass
    return result

# ══════════════════════════════════════════════════════════════
#  OPTIONS
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=1800)
def get_options_data(ticker):
    result = {"iv":None,"iv_percentile":None,"next_exp":None,"has_options":False,"error":None}
    try:
        stock = yf.Ticker(ticker)
        exps  = stock.options
        if not exps:
            result["error"] = "No options chain available"; return result
        result.update({"has_options":True,"next_exp":exps[0]})
        chain = stock.option_chain(exps[0])
        calls = chain.calls
        if "impliedVolatility" in calls.columns and not calls.empty:
            iv_vals = calls["impliedVolatility"].dropna()
            result["iv"] = float(iv_vals.median()) * 100
    except Exception as e:
        result["error"] = str(e)
    return result

def options_recommendation(trend_bias, event_risk, iv, rsi_val, atr_pct):
    if event_risk=="High":
        return "⚠️ Avoid / Wait", "Earnings ≤5 days — IV inflated, direction binary. Wait for post-earnings."
    if iv is None:
        if trend_bias=="Bullish" and rsi_val<65:
            return "📈 Stock / Calls", "No IV data. Trend bullish — stock or ATM calls OK."
        elif trend_bias=="Bearish":
            return "📉 Puts / Short", "No IV data. Trend bearish — puts or short bias."
        return "⏳ Wait", "Trend unclear. Insufficient data."
    if iv>60:
        if trend_bias=="Bullish":
            return "🐂 Bull Call Spread", f"IV={iv:.1f}% elevated. Spread limits debit cost while expressing bull view."
        elif trend_bias=="Bearish":
            return "🐻 Bear Put Spread", f"IV={iv:.1f}% elevated. Spread keeps premium cost manageable."
        return "⏳ Wait", f"IV={iv:.1f}% elevated but trend unclear. Not enough conviction."
    if iv<=30:
        if trend_bias=="Bullish" and rsi_val<65:
            return "📈 Calls (ATM/slight ITM)", f"Low IV={iv:.1f}% — options cheap for directional bets."
        elif trend_bias=="Bearish":
            return "📉 Puts (ATM)", f"Low IV={iv:.1f}% — cheap puts for downside."
    if trend_bias=="Bullish" and rsi_val<65:
        return "📈 Calls", f"Moderate IV={iv:.1f}%, bullish trend, RSI healthy."
    elif trend_bias=="Bearish":
        return "📉 Puts", f"Moderate IV={iv:.1f}%, bearish trend."
    if atr_pct>4:
        return "🔄 Stock Only", f"High ATR={atr_pct:.1f}% makes options expensive. Prefer stock."
    return "⏳ Wait", "Setup not clear enough for options."

# ══════════════════════════════════════════════════════════════
#  SIGNAL CHIP GENERATOR
# ══════════════════════════════════════════════════════════════
def generate_signals(df, a, catalyst, mtf_align, info=None):
    """Return list of (label, color) signal chips."""
    chips = []
    price    = a["price"]
    rsi_val  = a["rsi"]
    ema20    = a["ema20"]
    ema50    = a["ema50"]
    ema200   = a["ema200"]
    st_dir   = str(df.iloc[-1]["SupertrendDir"])
    vol      = float(df["Volume"].iloc[-1])
    vol_avg  = float(df["Vol_EMA20"].iloc[-1]) if "Vol_EMA20" in df.columns else vol
    vol_ratio= vol / vol_avg if vol_avg>0 else 1.0
    atr_pct  = a["atr_pct"]
    bb_pos   = a["bb_pos"]
    hi52     = float(df["High"].tail(252).max()) if len(df)>=252 else float(df["High"].max())
    lo52     = float(df["Low"].tail(252).min())  if len(df)>=252 else float(df["Low"].min())
    pct_from_hi = (price - hi52) / hi52 * 100
    pct_from_lo = (price - lo52) / lo52 * 100

    # Trend signals
    if price > ema20 > ema50 > ema200 and st_dir=="up":
        chips.append(("✅ Perfect Bull Stack","green"))
    elif price > ema50 and price > ema200 and st_dir=="up":
        chips.append(("✅ Above All MAs","green"))
    elif price < ema50 and price < ema200:
        chips.append(("⚠️ Below Key MAs","red"))

    # 52-week proximity
    if pct_from_hi > -5:
        chips.append(("📈 Near 52W High","green"))
    elif pct_from_lo < 15:
        chips.append(("📉 Near 52W Low","red"))

    # RSI
    if rsi_val > 70:
        chips.append(("⚠️ RSI Overbought","gold"))
    elif rsi_val < 30:
        chips.append(("⚠️ RSI Oversold","red"))
    elif 45 < rsi_val < 65:
        chips.append(("✅ RSI Healthy","green"))

    # Volume
    if vol_ratio > 1.5:
        chips.append((f"🔊 Volume {vol_ratio:.1f}x Avg","blue"))
    elif vol_ratio < 0.5:
        chips.append(("🔇 Low Volume","gold"))

    # Bollinger
    if bb_pos > 85:
        chips.append(("⚡ Upper BB — Extended","gold"))
    elif bb_pos < 15:
        chips.append(("🎯 Lower BB — Oversold Zone","green"))

    # MTF alignment
    if mtf_align >= 70:
        chips.append((f"🎯 MTF Aligned {mtf_align}/100","green"))
    elif mtf_align <= 35:
        chips.append((f"⚔️ MTF Conflict {mtf_align}/100","red"))

    # Catalyst
    dte = catalyst.get("days_to_earnings")
    if dte is not None and dte <= 5:
        chips.append((f"🔴 Earnings in {dte}d","red"))
    elif dte is not None and dte <= 14:
        chips.append((f"🟡 Earnings in {dte}d","gold"))

    # Supertrend
    if st_dir == "up":
        chips.append(("🟢 Supertrend UP","green"))
    else:
        chips.append(("🔴 Supertrend DOWN","red"))

    # MACD
    macd_v = float(df.iloc[-1]["MACD"])
    macd_s = float(df.iloc[-1]["MACD_Signal"])
    macd_h_prev = float(df.iloc[-2]["MACD_Hist"]) if len(df)>1 else 0
    macd_h_curr = float(df.iloc[-1]["MACD_Hist"])
    if macd_v > macd_s and macd_h_curr > 0 and macd_h_curr > macd_h_prev:
        chips.append(("📈 MACD Bullish Momentum","green"))
    elif macd_v < macd_s and macd_h_curr < 0 and macd_h_curr < macd_h_prev:
        chips.append(("📉 MACD Bearish Momentum","red"))
    elif macd_v > macd_s:
        chips.append(("📊 MACD Above Signal","green"))
    else:
        chips.append(("📊 MACD Below Signal","red"))

    # Institutional (if available)
    if info:
        inst = info.get("heldPercentInstitutions")
        if inst and inst > 0.7:
            chips.append((f"🏛️ {inst*100:.0f}% Institutional","blue"))

    return chips

# ══════════════════════════════════════════════════════════════
#  MULTI-TIMEFRAME ENGINE
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def get_mtf_data(ticker):
    timeframes  = ["5m","15m","1h","4h","1d","1wk"]
    rows        = []
    bull_total  = 0.0
    tf_count    = 0
    for tf in timeframes:
        df_tf,_,err,_ = get_stock_data(ticker,tf)
        if err or df_tf is None or len(df_tf)<20:
            rows.append({"TF":tf,"EMA9":"—","EMA20":"—","RSI":"—","MACD":"—","ST":"—","Verdict":"⬜ No Data"})
            continue
        lat  = df_tf.iloc[-1]
        p,e9,e20 = float(lat["Close"]),float(lat["EMA9"]),float(lat["EMA20"])
        r,mc,ms  = float(lat["RSI"]),float(lat["MACD"]),float(lat["MACD_Signal"])
        sd       = str(lat["SupertrendDir"])
        vw       = float(lat["VWAP"]) if "VWAP" in df_tf.columns else None
        bull = 0
        if p>e9:  bull+=1
        if p>e20: bull+=1
        if mc>ms: bull+=1
        if sd=="up": bull+=1
        if r>50:  bull+=1
        if vw and p>vw: bull+=1
        denom = 6
        pct = bull/denom
        if pct>=0.75:   v="🟢 Bullish"; bull_total+=1.0
        elif pct>=0.5:  v="🟡 Mild Bull"; bull_total+=0.5
        elif pct>=0.25: v="🟠 Mild Bear"
        else:           v="🔴 Bearish"
        tf_count+=1
        rows.append({"TF":tf,"EMA9":"▲" if p>e9 else "▼","EMA20":"▲" if p>e20 else "▼",
                     "RSI":f"{r:.0f}","MACD":"▲" if mc>ms else "▼",
                     "ST":"▲" if sd=="up" else "▼","Verdict":v})
    align = int((bull_total/max(tf_count,1))*100)
    if align>=75:
        pb="Strong Bullish"; trig="Buy pullbacks to EMA9/20"; cf="Watch overbought RSI on 1d"
    elif align>=55:
        pb="Mild Bullish"; trig="Wait for 1h close above EMA20"; cf="Size down — some TF conflict"
    elif align>=45:
        pb="Neutral/Choppy"; trig="Avoid — wait for resolution"; cf="High whipsaw risk"
    elif align>=25:
        pb="Mild Bearish"; trig="Avoid longs; watch bounces for shorts"; cf="1d may still transition"
    else:
        pb="Strong Bearish"; trig="Short on failed bounces to EMA20"; cf="Confirm with volume"
    return {"rows":rows,"alignment_score":align,"primary_bias":pb,"trigger":trig,"conflict_note":cf}

# ══════════════════════════════════════════════════════════════
#  SCORING ENGINE — point-based, fully differentiated
# ══════════════════════════════════════════════════════════════
def score_fundamentals(info):
    pts = 0.0
    rev_g  = info.get("revenueGrowth")
    gm     = info.get("grossMargins")
    nm     = info.get("profitMargins")
    op_m   = info.get("operatingMargins")
    pe     = info.get("trailingPE")
    fwd_pe = info.get("forwardPE")
    peg    = info.get("pegRatio")
    eps    = info.get("trailingEps",0) or 0
    debt   = info.get("totalDebt",0) or 0
    cash   = info.get("totalCash",0) or 0
    fcf    = info.get("freeCashflow",0) or 0
    roe    = info.get("returnOnEquity",0) or 0
    eps_g  = info.get("earningsGrowth",0) or 0

    # Revenue growth
    if rev_g is not None:
        if rev_g>0.25:   pts+=1.0
        elif rev_g>0.15: pts+=0.75
        elif rev_g>0.05: pts+=0.5
        elif rev_g>0:    pts+=0.25
        else:            pts-=0.5

    # Gross margin
    if gm is not None:
        if gm>0.65:   pts+=1.0
        elif gm>0.50: pts+=0.75
        elif gm>0.35: pts+=0.5
        elif gm>0.20: pts+=0.25
        else:         pts-=0.25

    # Net margin
    if nm is not None:
        if nm>0.25:   pts+=0.75
        elif nm>0.15: pts+=0.5
        elif nm>0.05: pts+=0.25
        elif nm<0:    pts-=0.75

    # EPS growth (quarterly)
    if eps_g>0.20:   pts+=0.5
    elif eps_g>0.10: pts+=0.25
    elif eps_g<-0.10: pts-=0.25

    # FCF
    if fcf>0: pts+=0.5
    elif fcf<0: pts-=0.25

    # Balance sheet
    net_cash = cash - debt
    if net_cash>0: pts+=0.5
    else: pts-=0.25

    # P/E
    if pe is not None and pe>0:
        if pe<18:   pts+=0.5
        elif pe<30: pts+=0.25
        elif pe>70: pts-=0.5

    # ROE
    if roe>0.25: pts+=0.5
    elif roe>0.15: pts+=0.25
    elif roe<0: pts-=0.25

    return max(1.0, min(5.0, round(pts, 1)))

def score_technicals(df):
    if df is None or len(df)<20:
        return 3.0, "Neutral", {}
    lat    = df.iloc[-1]
    price  = float(lat["Close"])
    ema9   = float(lat["EMA9"])
    ema20  = float(lat["EMA20"])
    ema50  = float(lat["EMA50"])
    ema200 = float(lat["EMA200"])
    rsi_v  = float(lat["RSI"])
    macd_v = float(lat["MACD"])
    macd_s = float(lat["MACD_Signal"])
    st_dir = str(lat["SupertrendDir"])
    vwap_v = float(lat["VWAP"])
    vol    = float(lat["Volume"])
    vol_avg= float(lat.get("Vol_EMA20", vol))

    pts = 0.0
    if price>ema200: pts+=1.0
    if price>ema50:  pts+=0.75
    if price>ema20:  pts+=0.5
    if price>ema9:   pts+=0.25
    if st_dir=="up": pts+=0.75
    if macd_v>macd_s: pts+=0.5
    if 45<rsi_v<65:  pts+=0.5
    elif 65<=rsi_v<75: pts+=0.15
    elif rsi_v>=75:  pts-=0.25
    elif rsi_v<30:   pts-=0.5
    if price>vwap_v: pts+=0.25
    if vol_avg>0 and vol>vol_avg*1.2: pts+=0.25

    # Check for EMA alignment (adds bonus)
    if ema9>ema20>ema50>ema200 and price>ema9: pts+=0.5   # perfect stack bonus

    trend = "Bullish" if pts>=2.5 else ("Bearish" if pts<1.2 else "Neutral")
    return max(1.0, min(5.0, round(pts, 1))), trend, {
        "price":price,"ema9":ema9,"ema20":ema20,"ema50":ema50,"ema200":ema200,
        "rsi":rsi_v,"macd":macd_v,"macd_sig":macd_s,"st_dir":st_dir,
        "vwap":vwap_v,"vol_ratio":vol/vol_avg if vol_avg>0 else 1.0
    }

# ══════════════════════════════════════════════════════════════
#  FULL ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════
def analyze(df, info, ticker, sentiment_score):
    lat    = df.iloc[-1]
    price  = float(lat["Close"])
    ema9   = float(lat["EMA9"])
    ema20  = float(lat["EMA20"])
    ema50  = float(lat["EMA50"])
    ema200 = float(lat["EMA200"])
    rsi_v  = float(lat["RSI"])
    vwap_v = float(lat["VWAP"])
    st_val = float(lat["Supertrend"])
    st_dir = str(lat["SupertrendDir"])
    atr_v  = float(lat["ATR"]) if not np.isnan(float(lat["ATR"])) else price*0.02
    macd_v = float(lat["MACD"])
    macd_s = float(lat["MACD_Signal"])
    bb_u   = float(lat["BB_Upper"])
    bb_l   = float(lat["BB_Lower"])
    bb_m   = float(lat["BB_Mid"])
    vol    = float(lat["Volume"])
    vol_avg= float(lat.get("Vol_EMA20", vol))
    bb_pos = ((price-bb_l)/(bb_u-bb_l)*100) if (bb_u-bb_l)>0 else 50

    s1,s2,s3,r1,r2,r3 = get_key_levels(df)

    # Scoring
    fund_score = score_fundamentals(info)
    tech_score, trend_bias, tech_vals = score_technicals(df)

    # Risk score
    daily_ret  = df["Close"].pct_change().dropna()
    vol_daily  = float(daily_ret.std())*100
    max_dd     = float(((df["Close"]/df["Close"].cummax())-1).min())*100
    if vol_daily<1.0:   risk_score=4.5
    elif vol_daily<2.0: risk_score=3.5
    elif vol_daily<3.0: risk_score=3.0
    elif vol_daily<4.0: risk_score=2.5
    else:               risk_score=2.0
    if max_dd<-40: risk_score=max(1.5, risk_score-0.5)

    # Plan score
    if trend_bias=="Bullish" and sentiment_score>=55:  plan_score=4.0
    elif trend_bias=="Bullish":                        plan_score=3.5
    elif trend_bias=="Bearish" and sentiment_score<40: plan_score=1.5
    elif trend_bias=="Bearish":                        plan_score=2.0
    else:                                              plan_score=2.5

    # Entry/Exit levels
    if trend_bias=="Bullish":
        entry_low  = min(ema20, ema50)*0.999
        entry_high = min(price*1.005, ema20*1.01)
        stop_loss  = max(s1 - atr_v*0.5, price - atr_v*2.0)
    else:
        entry_low  = price*0.99
        entry_high = price*1.005
        stop_loss  = price - atr_v*1.5

    risk_per_sh = max(price - stop_loss, atr_v*0.5)
    target_1    = price + risk_per_sh*2.0
    target_2    = price + risk_per_sh*3.5
    rr_1        = (target_1-price)/risk_per_sh if risk_per_sh>0 else 0

    ee_score = 4.0 if (trend_bias=="Bullish" and rr_1>=2.0) else (3.0 if rr_1>=1.5 else 2.0)

    # Mindset score
    if vol_daily>3.5:  mind_score=2.5
    elif vol_daily>2.0: mind_score=3.5
    else:              mind_score=4.5

    scores = {"Fundamentals":fund_score,"Technicals":tech_score,
              "Risk":risk_score,"Plan":plan_score,
              "Entry/Exit":ee_score,"Mindset":mind_score}
    overall = round(sum(scores.values())/len(scores), 2)

    # Key metrics for display
    rev_growth  = info.get("revenueGrowth",0) or 0
    eps_growth  = info.get("earningsGrowth",0) or 0
    inst_pct    = info.get("heldPercentInstitutions",0) or 0
    float_shares= info.get("floatShares",0) or 0
    vol_ratio   = vol/vol_avg if vol_avg>0 else 1.0

    return {
        "price":price,"rsi":rsi_v,"trend_bias":trend_bias,
        "atr":atr_v,"atr_pct":atr_v/price*100,
        "vol_daily":vol_daily,"support_watch":s1,"resistance_watch":r1,
        "entry_low":entry_low,"entry_high":entry_high,
        "stop_loss":stop_loss,"target_1":target_1,"target_2":target_2,"rr":rr_1,
        "scores":scores,"overall_score":overall,
        "levels":(s1,s2,s3,r1,r2,r3),
        "market_cap":info.get("marketCap"),"sector":info.get("sector",""),
        "ema9":ema9,"ema20":ema20,"ema50":ema50,"ema200":ema200,
        "vwap":vwap_v,"bb_pos":bb_pos,"bb_u":bb_u,"bb_l":bb_l,"bb_m":bb_m,
        "st_dir":st_dir,"st_val":st_val,"max_dd":max_dd,
        "rev_growth":rev_growth,"eps_growth":eps_growth,
        "inst_pct":inst_pct,"float_shares":float_shares,"vol_ratio":vol_ratio,
        "pe":info.get("trailingPE"),"fwd_pe":info.get("forwardPE"),
        "analyst_target":info.get("targetMeanPrice"),
        "fund_score":fund_score,"tech_score":tech_score,"risk_score":risk_score,
    }

# ══════════════════════════════════════════════════════════════
#  EXPERT AI COMMENTARY (OpenRouter)
# ══════════════════════════════════════════════════════════════
def get_openrouter_key():
    try: return st.secrets.get("OPENROUTER_API_KEY")
    except Exception: return None

def generate_ai_commentary(payload, provider="none", model="openrouter/auto"):
    if provider=="none":
        return _expert_python_summary(payload), None
    api_key = get_openrouter_key()
    if not api_key:
        return _expert_python_summary(payload), "⚠️ OPENROUTER_API_KEY not set. Showing Python analysis."

    ticker   = payload.get("ticker","?")
    price    = payload.get("price",0)
    trend    = payload.get("trend_bias","Neutral")
    rsi      = payload.get("rsi",50)
    align    = payload.get("alignment_score",50)
    iv       = payload.get("iv","N/A")
    sent_sc  = payload.get("sentiment_score",50)
    sent_lb  = payload.get("sentiment_label","Neutral")
    event    = payload.get("event_risk","Low")
    vehicle  = payload.get("options_vehicle","Wait")
    s1       = payload.get("support",price*0.95)
    r1       = payload.get("resistance",price*1.05)
    fund_sc  = payload.get("fund_score",3)
    tech_sc  = payload.get("tech_score",3)
    overall  = payload.get("overall_score",3)
    atr_pct  = payload.get("atr_pct",2)
    rev_g    = payload.get("rev_growth",0)
    eps_g    = payload.get("eps_growth",0)
    inst_pct = payload.get("inst_pct",0)
    max_dd   = payload.get("max_dd",-20)
    mtf      = payload.get("primary_bias","Neutral")

    prompt = f"""You are a senior portfolio manager and technical analyst with 20+ years experience.
Provide a comprehensive, actionable trading analysis for {ticker}.

== CURRENT DATA ==
Price: ${price:.2f} | Trend: {trend} | RSI: {rsi:.1f} | MTF Bias: {mtf}
MTF Alignment: {align}/100 | Overall Score: {overall}/5
Fundamental Score: {fund_sc}/5 | Technical Score: {tech_sc}/5
Revenue Growth: {rev_g*100:.1f}% | EPS Growth: {eps_g*100:.1f}%
Institutional Ownership: {inst_pct*100:.1f}%
Event Risk: {event} | IV: {iv if isinstance(iv,str) else f"{iv:.1f}%"}
News Sentiment: {sent_lb} ({sent_sc}/100)
ATR%: {atr_pct:.2f}% | Max Drawdown: {max_dd:.1f}%
Key Support: ${s1:.2f} | Key Resistance: ${r1:.2f}
Entry Zone: ${payload.get('entry_low',0):.2f}–${payload.get('entry_high',0):.2f}
Stop Loss: ${payload.get('stop_loss',0):.2f} | T1: ${payload.get('target_1',0):.2f} | T2: ${payload.get('target_2',0):.2f}
R:R: {payload.get('rr',0):.2f}x | Options Vehicle: {vehicle}

Respond with EXACTLY these 8 sections. Be specific and data-driven. Minimum 3 sentences per section.

### 📊 Executive Summary
[Overall assessment: bullish/bearish/neutral, why, key conviction level, actionability score]

### 🔬 Technical Analysis
[Price structure, EMA stack, trend confirmation, MACD reading, volume analysis, key pattern if any]

### 💼 Fundamental Assessment  
[Revenue/EPS growth quality, margin trends, institutional conviction, valuation vs peers, balance sheet]

### 🌐 Social & Market Sentiment
[News sentiment analysis, institutional buying/selling signals, sector momentum, macro tailwinds/headwinds, Reddit/retail sentiment inference from volatility]

### ⚡ Risk Assessment
[ATR-based risk, earnings proximity, max drawdown context, position sizing guidance, correlation risk]

### 🎯 Specific Trade Recommendation
[Exact entry trigger (price level or condition), stop loss rationale, T1 and T2 exits, position size % of portfolio, time horizon]

### 🚨 Key Catalysts & Invalidation
[2-3 specific upcoming catalysts, exact price that invalidates the thesis, what you'd need to see to change view]

### 🧠 Trader Psychology Note
[Common emotional traps for this specific setup, discipline rule to follow, one contrarian consideration]"""

    try:
        headers = {"Authorization":f"Bearer {api_key}","Content-Type":"application/json",
                   "HTTP-Referer":"https://utpal-trading-dashboard.streamlit.app","X-Title":"Utpal Trading"}
        body    = {"model":model,"messages":[{"role":"user","content":prompt}],"max_tokens":1200}
        resp    = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                headers=headers,json=body,timeout=30)
        if resp.status_code==200:
            return resp.json()["choices"][0]["message"]["content"], None
        return _expert_python_summary(payload), f"⚠️ OpenRouter {resp.status_code}. Python analysis shown."
    except Exception as e:
        return _expert_python_summary(payload), f"⚠️ API error: {str(e)[:80]}"

def _expert_python_summary(p):
    ticker   = p.get("ticker","?")
    trend    = p.get("trend_bias","Neutral")
    rsi      = p.get("rsi",50)
    align    = p.get("alignment_score",50)
    score    = p.get("overall_score",3)
    s1       = p.get("support",0)
    r1       = p.get("resistance",0)
    stop     = p.get("stop_loss",0)
    t1       = p.get("target_1",0)
    t2       = p.get("target_2",0)
    rr       = p.get("rr",0)
    event    = p.get("event_risk","Low")
    vehicle  = p.get("options_vehicle","Wait")
    sent_sc  = p.get("sentiment_score",50)
    sent_lb  = p.get("sentiment_label","Neutral")
    atr_pct  = p.get("atr_pct",2)
    rev_g    = p.get("rev_growth",0)*100
    eps_g    = p.get("eps_growth",0)*100
    fund_sc  = p.get("fund_score",3)
    tech_sc  = p.get("tech_score",3)
    inst_pct = p.get("inst_pct",0)*100
    mtf      = p.get("primary_bias","Neutral")
    price    = p.get("price",0)
    max_dd   = p.get("max_dd",-20)

    conviction = "HIGH" if score>=4 else ("MODERATE" if score>=3 else "LOW")
    trend_color = "bullish" if trend=="Bullish" else ("bearish" if trend=="Bearish" else "neutral")

    sections = []
    sections.append(f"""### 📊 Executive Summary
**{ticker}** presents a **{conviction} conviction {trend_color}** setup with an overall score of **{score}/5**.
Multi-timeframe alignment stands at **{align}/100** ({mtf}), suggesting {'strong directional conviction across timeframes' if align>=65 else ('mixed signals requiring patience' if align>=45 else 'significant trend conflict — caution advised')}.
{'News sentiment is ' + sent_lb.lower() + f' at {sent_sc}/100, providing ' + ('tailwind' if sent_sc>60 else ('headwind' if sent_sc<40 else 'neutral backdrop')) + ' for the trade thesis.' if True else ''}
Actionability: {'✅ Ready to trade with defined risk' if score>=3.5 and align>=55 and event!='High' else '⚠️ Wait for better setup or lower risk environment'}.""")

    ema9=p.get("ema9",price*0.99); ema20=p.get("ema20",price*0.98); ema50=p.get("ema50",price*0.95); ema200=p.get("ema200",price*0.90)
    sections.append(f"""### 🔬 Technical Analysis
Price at **${price:.2f}** is {'above' if price>ema20 else 'below'} EMA20 (${ema20:.2f}) and {'above' if price>ema50 else 'below'} EMA50 (${ema50:.2f}), {'suggesting continuation of the uptrend' if price>ema20>ema50 else 'indicating weakness in the trend structure'}.
RSI at **{rsi:.1f}** is {'in the healthy 45–65 bullish zone — room to run without being extended' if 45<rsi<65 else ('overbought — short-term pullback risk elevated' if rsi>70 else ('oversold — potential mean reversion bounce' if rsi<30 else 'neutral — watching for directional confirmation'))}.
Technical score **{tech_sc}/5**: VWAP {'above' if price>p.get('vwap',price) else 'below'} price (buyers {'in' if price>p.get('vwap',price) else 'losing'} control). 
Key support at **${s1:.2f}**, resistance at **${r1:.2f}** — defined range of **{((r1-s1)/s1*100):.1f}%**.""")

    sections.append(f"""### 💼 Fundamental Assessment
Fundamental score **{fund_sc}/5**. Revenue growth at **{rev_g:.1f}%** YoY {'— above 15% threshold indicating strong demand expansion' if rev_g>15 else ('— moderate growth suggesting stable but not accelerating business' if rev_g>5 else '— growth concerns require watching next earnings for improvement')}.
EPS trajectory at **{eps_g:.1f}%** {'outpacing revenue growth — expanding margins, quality earnings' if eps_g>rev_g else 'tracking revenue — inline margin performance'}.
Institutional ownership at **{inst_pct:.1f}%** {'suggests strong smart money conviction — reduces supply overhang' if inst_pct>65 else ('moderate institutional interest' if inst_pct>40 else '— lower institutional involvement, higher retail-driven volatility risk')}.
Valuation context: {'P/E of ' + str(round(p.get("pe",0) or 0,1)) + 'x — ' + ('premium priced for growth, needs execution' if (p.get("pe") or 0)>35 else 'reasonable relative to growth rate') if p.get("pe") else 'P/E data unavailable — use EV/Sales or price/FCF for context'}.""")

    sections.append(f"""### 🌐 Social & Market Sentiment
News sentiment score **{sent_sc}/100** ({sent_lb}) based on aggregated headlines from Google News and Yahoo Finance across earnings, analyst actions, and sector news.
{'Bullish catalysts detected: analyst upgrades, positive guidance references, or beat expectations language in recent coverage.' if sent_sc>60 else ('Bearish language detected: downgrade risks, miss concerns, or macro headwinds referenced in news flow.' if sent_sc<40 else 'Sentiment is balanced — no strong directional bias from news flow currently.')}
Implied volatility context: {'IV elevated — suggests options market pricing in significant move, consistent with news-driven uncertainty' if isinstance(p.get("iv"),float) and p.get("iv",0)>40 else ('IV muted — market not pricing large move, options cheap for directional bets' if isinstance(p.get("iv"),float) else 'IV data unavailable')}.
Sector momentum: {'tech/semiconductor names showing correlation to AI capex cycle — macro tailwind intact' if 'Tech' in (p.get("sector","") or '') or ticker in ['AVGO','NVDA','AMD','INTC','MU'] else 'monitor sector rotation for any headwinds'}.""")

    sections.append(f"""### ⚡ Risk Assessment
ATR-based daily expected move: **{atr_pct:.2f}%** of price. {'High volatility — size down by 30–50% vs normal position' if atr_pct>3 else ('Moderate volatility — standard ATR-based stops appropriate' if atr_pct>1.5 else 'Low volatility — wider stops less likely to trigger on noise')}.
Event risk: **{event}** {'— earnings binary risk, IV inflated, recommend avoiding new positions or using spreads to cap debit' if event=='High' else ('— earnings approaching, reduce size and use defined risk structures' if event=='Moderate' else '— clean window with no known binary events, favorable environment')}.
Historical max drawdown: **{max_dd:.1f}%** — size positions so max loss on this trade ≤ 1–2% of total portfolio.
Stop at **${stop:.2f}** — {'this is {:.1f}% below current price, consistent with ATR-based noise floor'.format((price-stop)/price*100) if price>0 else 'defined below key support'}. Honoring the stop is non-negotiable.""")

    sections.append(f"""### 🎯 Specific Trade Recommendation
**Direction:** {'Long (Buy)' if trend=='Bullish' else ('Short (Sell)' if trend=='Bearish' else 'Wait for confirmation')}
**Entry trigger:** {'Pullback to EMA20 (${:.2f}) on declining volume, then green candle confirmation'.format(ema20) if trend=='Bullish' else ('Bounce to EMA20 (${:.2f}) on low volume, then red candle rejection'.format(ema20) if trend=='Bearish' else 'Wait for price to reclaim EMA50 with volume before entry')}
**Position sizing:** Risk max 1% of account — with stop at ${stop:.2f}, that sets your size automatically via position sizer below.
**Vehicle:** {vehicle} | **Entry zone:** ${p.get('entry_low',0):.2f}–${p.get('entry_high',0):.2f} | **Stop:** ${stop:.2f} | **T1:** ${t1:.2f} (+{((t1-price)/price*100):.1f}%) | **T2:** ${t2:.2f} (+{((t2-price)/price*100):.1f}%)
**R:R:** **{rr:.2f}x** {'✅ Acceptable' if rr>=2 else '⚠️ Below 2x minimum — improve entry or skip'}. Time horizon: 2–6 weeks for swing trade.""")

    sections.append(f"""### 🚨 Key Catalysts & Invalidation
**Upcoming catalysts to watch:** {'Earnings in ' + str(p.get("days_to_earnings","?")) + ' days — primary binary event' if p.get("days_to_earnings") else 'Next earnings release'}, Federal Reserve rate decisions, sector earnings from peers (watch for read-across), macro data (CPI/NFP).
**Invalidation:** Trade is invalidated on **close below ${stop:.2f}**. If this level breaks on high volume, the support structure is damaged — exit without hesitation, no averaging down.
**Bull case:** {trend=='Bullish' and f'Price reclaims and holds ${r1:.2f} resistance on volume → confirms breakout, add to position' or f'Price reclaims EMA50 (${ema50:.2f}) → first sign of recovery'}. 
**Bear case:** {'Close below EMA200 (${:.2f}) invalidates entire bullish thesis'.format(ema200)} — signals structural breakdown, not just a pullback.""")

    sections.append(f"""### 🧠 Trader Psychology Note
**Primary emotional trap here:** {'Chasing — if you missed the initial move, do NOT chase above resistance. Wait for the next pullback to EMA20.' if trend=='Bullish' and rsi>65 else ('Catching a falling knife — every bounce looks like a reversal in a downtrend. Need structure change, not just a green day.' if trend=='Bearish' else 'Analysis paralysis — waiting for perfect setup means missing good setups. Define your criteria and execute.')}
**Discipline rule:** Write your entry, stop, and target BEFORE placing the order. If price doesn't reach your entry zone, you don't trade — that IS a trade decision.
**Contrarian consideration:** {'At RSI ' + f'{rsi:.0f}' + ', the crowd is ' + ('euphoric — consider taking partial profits at T1 rather than holding for T2' if rsi>70 else ('fearful — which historically is where swing trades are built' if rsi<40 else 'neutral — no extreme to fade')) + '. Where is the majority wrong here?'}.""")

    return "\n\n".join(sections)

# ══════════════════════════════════════════════════════════════
#  AI PORTFOLIO STRATEGY GENERATOR
# ══════════════════════════════════════════════════════════════
def generate_portfolio_strategy(positions_data, watchlist_signals, market_regime, provider="none", model="openrouter/auto"):
    portfolio_summary = "\n".join([
        f"- {t}: {d['shares']} shares @ ${d['avg_cost']:.4f} CAD, current ${d.get('current',d['avg_cost']):.2f}, P&L {d.get('pnl_pct',0):+.1f}%"
        for t,d in positions_data.items()
    ])
    watchlist_summary = "\n".join([
        f"- {s['ticker']}: {s['trend']} | RSI {s['rsi']:.0f} | ST {s['st_dir']} | Align {s['align']}/100 | Signal: {s['signal']}"
        for s in watchlist_signals[:8]
    ])
    if provider=="none" or not get_openrouter_key():
        # Python-generated strategy
        strong = [s for s in watchlist_signals if s.get("align",0)>=65 and s.get("st_dir")=="up"]
        risky  = [s for s in watchlist_signals if s.get("align",0)<=35]
        lines  = [f"**Market Regime:** {market_regime.get('regime','Unknown')} | SPY: {market_regime.get('SPY','?')} | VIX: {market_regime.get('VIX','?')}"]
        if strong:
            lines.append(f"\n**🟢 Top Watchlist Opportunities ({len(strong)} setups):**")
            for s in strong[:3]:
                lines.append(f"- **{s['ticker']}**: MTF {s['align']}/100, Supertrend UP, RSI {s['rsi']:.0f} — consider adding on pullback to EMA20")
        if risky:
            lines.append(f"\n**🔴 Avoid for Now ({len(risky)} weak setups):** {', '.join(r['ticker'] for r in risky[:4])}")
        lines.append(f"\n**Portfolio Note:** With ~${CASH_USD:,.0f} USD cash, deploy in 3 tranches: 33% on confirmed signal, 33% on continuation, 34% reserve.")
        return "\n".join(lines)

    prompt = f"""You are a professional portfolio manager. Analyze this swing trading portfolio and provide specific strategic recommendations.

CURRENT POSITIONS:
{portfolio_summary}
Cash reserve: ~${CASH_USD:,.0f} USD (~${CASH_USD*CAD_RATE:,.0f} CAD)
Portfolio goal: ${GOAL_LOW:,}–${GOAL_HIGH:,} CAD

MARKET REGIME: {market_regime.get('regime','Unknown')} | SPY: {market_regime.get('SPY','?')} | QQQ: {market_regime.get('QQQ','?')} | VIX: {market_regime.get('VIX','N/A')}

TOP WATCHLIST SIGNALS:
{watchlist_summary}

Provide a structured strategy covering:
1. **Portfolio Assessment** — current position health, P&L status, concentration risk
2. **Top 3 Watchlist Opportunities** — specific entry triggers, position sizes as % of cash
3. **Rotation Strategy** — when/how to rotate between AVGO and META or into new positions  
4. **Cash Deployment Plan** — how to deploy the ${CASH_USD:,.0f} USD across 3–4 tranches
5. **Key Risks to Monitor** — 3 specific risk scenarios with mitigation actions
6. **Target & Timeline** — path to ${GOAL_LOW:,}–${GOAL_HIGH:,} CAD goal

Be specific, actionable, and reference actual prices/percentages."""

    try:
        api_key = get_openrouter_key()
        headers = {"Authorization":f"Bearer {api_key}","Content-Type":"application/json",
                   "HTTP-Referer":"https://utpal-trading-dashboard.streamlit.app"}
        body    = {"model":model,"messages":[{"role":"user","content":prompt}],"max_tokens":1000}
        resp    = requests.post("https://openrouter.ai/api/v1/chat/completions",headers=headers,json=body,timeout=30)
        if resp.status_code==200:
            return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        pass
    return generate_portfolio_strategy(positions_data, watchlist_signals, market_regime, "none")

# ══════════════════════════════════════════════════════════════
#  CHART — fixed rangebreaks, improved layout
# ══════════════════════════════════════════════════════════════
def build_chart(df, ticker, interval, show_bb=True, show_vwap=True, show_st=True,
                show_ema=True, show_macd=False, range_days=None):
    if range_days:
        df = df.tail(range_days)

    vol_colors = np.where(df["Close"] >= df["Open"], "#26a69a", "#ef5350")
    rows    = 3 if show_macd else 2
    heights = [0.60,0.15,0.25] if show_macd else [0.74,0.26]

    fig = make_subplots(rows=rows,cols=1,shared_xaxes=True,
                        vertical_spacing=0.01,row_heights=heights)

    fig.add_trace(go.Candlestick(
        x=df.index,open=df["Open"],high=df["High"],low=df["Low"],close=df["Close"],
        name="Price",
        increasing=dict(line=dict(color="#26a69a",width=1),fillcolor="#26a69a"),
        decreasing=dict(line=dict(color="#ef5350",width=1),fillcolor="#ef5350"),
        whiskerwidth=0,
    ),row=1,col=1)

    if show_bb:
        fig.add_trace(go.Scatter(x=df.index,y=df["BB_Upper"],name="BB Upper",
            line=dict(color="#5c6bc0",width=0.8,dash="dot"),showlegend=False),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["BB_Lower"],name="BB Lower",
            line=dict(color="#5c6bc0",width=0.8,dash="dot"),
            fill="tonexty",fillcolor="rgba(92,107,192,0.05)",showlegend=False),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["BB_Mid"],name="BB Mid",
            line=dict(color="#5c6bc0",width=0.7,dash="dash"),showlegend=False),row=1,col=1)

    if show_ema:
        for col_n,color,lw,label in [
            ("EMA9","#ffffff",1.0,"EMA 9"),("EMA20","#f5c542",1.0,"EMA 20"),
            ("EMA50","#4da3ff",1.2,"EMA 50"),("EMA200","#b388ff",1.4,"EMA 200"),
        ]:
            fig.add_trace(go.Scatter(x=df.index,y=df[col_n],mode="lines",
                name=label,line=dict(color=color,width=lw)),row=1,col=1)

    if show_vwap:
        fig.add_trace(go.Scatter(x=df.index,y=df["VWAP"],mode="lines",name="VWAP",
            line=dict(color="#ff9800",width=1.5,dash="dash")),row=1,col=1)

    if show_st:
        st_up   = df["Supertrend"].where(df["SupertrendDir"]=="up")
        st_down = df["Supertrend"].where(df["SupertrendDir"]=="down")
        fig.add_trace(go.Scatter(x=df.index,y=st_up,mode="lines",name="ST Bull",
            line=dict(color="#26a69a",width=2)),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=st_down,mode="lines",name="ST Bear",
            line=dict(color="#ef5350",width=2)),row=1,col=1)

    s1,s2,s3,r1,r2,r3 = get_key_levels(df)
    price_now = float(df["Close"].iloc[-1])
    for level,color,label in [(s1,"#26a69a","S1"),(r1,"#ef5350","R1"),(s2,"#1a7a72","S2"),(r2,"#a33534","R2")]:
        fig.add_hline(y=level,line_dash="dot",line_color=color,line_width=0.9,
                      annotation_text=f" {label}: ${level:.2f}",
                      annotation_font_color=color,annotation_font_size=10,row=1,col=1)

    fig.add_trace(go.Bar(x=df.index,y=df["Volume"],name="Volume",
        marker_color=vol_colors,showlegend=False),row=2,col=1)
    # Volume EMA line
    fig.add_trace(go.Scatter(x=df.index,y=df["Vol_EMA20"],mode="lines",name="Vol EMA20",
        line=dict(color="#f5c542",width=1.0,dash="dot"),showlegend=False),row=2,col=1)

    if show_macd:
        macd_colors = np.where(df["MACD_Hist"]>=0,"#26a69a","#ef5350")
        fig.add_trace(go.Bar(x=df.index,y=df["MACD_Hist"],name="MACD Hist",
            marker_color=macd_colors,showlegend=False),row=3,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["MACD"],mode="lines",name="MACD",
            line=dict(color="#2962ff",width=1.2)),row=3,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["MACD_Signal"],mode="lines",name="Signal",
            line=dict(color="#ff6b35",width=1.0)),row=3,col=1)
        fig.add_hline(y=0,line_color="#363a45",line_width=0.5,row=3,col=1)

    fig.update_layout(
        title=dict(text=f"  {ticker}",font=dict(size=16,color="#d1d4dc"),x=0),
        template="plotly_dark",
        height=700 if show_macd else 580,
        dragmode="pan",hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin=dict(l=50,r=10,t=36,b=10),
        legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="left",x=0,font=dict(size=11)),
        plot_bgcolor="#131722",paper_bgcolor="#131722",
        font=dict(color="#d1d4dc",size=11),
    )

    # ── RANGEBREAKS: remove weekends & non-market hours ──
    rangebreaks = [dict(bounds=["sat","mon"])]  # always remove weekends
    if interval in ("1m","5m","15m","1h"):
        rangebreaks.append(dict(bounds=[16,9.5],pattern="hour"))  # US market hours
    elif interval == "4h":
        rangebreaks.append(dict(bounds=[20,4],pattern="hour"))

    fig.update_xaxes(
        showgrid=True,gridcolor="#1f2937",gridwidth=0.5,
        zeroline=False,showspikes=True,spikemode="across",
        spikecolor="#787b86",spikethickness=1,
        showline=True,linecolor="#363a45",
        rangebreaks=rangebreaks,
        rangeselector=dict(
            buttons=[
                dict(count=5,label="5D",step="day",stepmode="backward"),
                dict(count=1,label="1M",step="month",stepmode="backward"),
                dict(count=3,label="3M",step="month",stepmode="backward"),
                dict(count=6,label="6M",step="month",stepmode="backward"),
                dict(count=1,label="1Y",step="year",stepmode="backward"),
                dict(step="all",label="All"),
            ],
            bgcolor="#1e222d",activecolor="#2962ff",
            font=dict(color="#d1d4dc",size=11),
            bordercolor="#363a45",borderwidth=1,
        ) if len(df)>20 else None,
    )
    fig.update_yaxes(showgrid=True,gridcolor="#1f2937",gridwidth=0.5,
                     zeroline=False,showline=True,linecolor="#363a45",
                     tickformat=".2f",side="right")
    fig.update_yaxes(title_text="",row=2,col=1,tickformat=".2s")
    if show_macd:
        fig.update_yaxes(title_text="MACD",row=3,col=1)
    return fig

# ══════════════════════════════════════════════════════════════
#  POSITION SIZING
# ══════════════════════════════════════════════════════════════
def calc_position_size(account, cash_avail, risk_pct, entry, stop, t1, t2, asset_type, atr_val=None):
    if entry<=0 or stop<=0 or stop>=entry:
        return None,"⚠️ Stop must be below entry and both > 0."
    max_risk = account*(risk_pct/100)
    risk_u   = entry - stop
    units    = max_risk/risk_u if risk_u>0 else 0
    label_u  = "shares"
    if asset_type=="Option":
        units = max(1, int(units/100)); label_u="contracts"; pos_val = units*100*entry
    else:
        units = int(units); pos_val = units*entry
    rr1 = (t1-entry)/risk_u if risk_u>0 and t1>entry else 0
    rr2 = (t2-entry)/risk_u if risk_u>0 and t2>entry else 0
    warnings = []
    if pos_val>cash_avail: warnings.append(f"⚠️ Position ${pos_val:,.0f} > available cash ${cash_avail:,.0f}. Scale down.")
    if atr_val and risk_u/atr_val<0.5: warnings.append(f"⚠️ Stop {risk_u/atr_val:.1f}x ATR — very tight, high noise risk.")
    if rr1>0 and rr1<1.5: warnings.append(f"⚠️ R:R to T1 = {rr1:.2f}x — below 1.5x minimum.")
    return {"max_risk":max_risk,"units":units,"label_unit":label_u,"pos_value":pos_val,
            "risk_per_unit":risk_u,"rr1":rr1,"rr2":rr2}, "\n".join(warnings) if warnings else None

# ══════════════════════════════════════════════════════════════
#  JOURNAL
# ══════════════════════════════════════════════════════════════
def load_journal():
    if not os.path.exists(JOURNAL_FILE): return pd.DataFrame(columns=JOURNAL_COLS)
    try:
        df = pd.read_csv(JOURNAL_FILE)
        for c in JOURNAL_COLS:
            if c not in df.columns: df[c]=""
        return df
    except Exception:
        return pd.DataFrame(columns=JOURNAL_COLS)

def save_trade(d):
    exists = os.path.exists(JOURNAL_FILE)
    with open(JOURNAL_FILE,"a",newline="") as f:
        w = csv.DictWriter(f,fieldnames=JOURNAL_COLS)
        if not exists: w.writeheader()
        w.writerow({c:d.get(c,"") for c in JOURNAL_COLS})

def journal_analytics(df):
    if df.empty or "result" not in df.columns: return {}
    df = df.copy()
    df["pnl"] = pd.to_numeric(df["pnl"],errors="coerce").fillna(0)
    df["alignment_score"] = pd.to_numeric(df["alignment_score"],errors="coerce").fillna(0)
    total = len(df)
    wins  = df[df["result"].str.lower()=="win"]
    losses= df[df["result"].str.lower()=="loss"]
    wr    = len(wins)/total*100 if total>0 else 0
    aw    = wins["pnl"].mean() if not wins.empty else 0
    al    = losses["pnl"].mean() if not losses.empty else 0
    gp    = wins["pnl"].sum()
    gl    = abs(losses["pnl"].sum())
    pf    = gp/gl if gl>0 else float("inf")
    exp   = (wr/100*aw)+((1-wr/100)*al)
    insights = []
    ha = df[df["alignment_score"]>=70]; la = df[df["alignment_score"]<50]
    if len(ha)>=2 and len(la)>=2:
        wrh = len(ha[ha["result"].str.lower()=="win"])/len(ha)*100
        wrl = len(la[la["result"].str.lower()=="win"])/len(la)*100
        if wrh>wrl+10:
            insights.append(f"✅ High-alignment trades win {wrh:.0f}% vs {wrl:.0f}% for low-alignment. Prioritize MTF alignment ≥70.")
    return {"total":total,"win_rate":wr,"avg_win":aw,"avg_loss":al,"total_pnl":df["pnl"].sum(),
            "profit_factor":pf,"expectancy":exp,"insights":insights}

# ══════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════
def fmt_large(n):
    if n is None: return "N/A"
    if n>=1e12: return f"${n/1e12:.2f}T"
    if n>=1e9:  return f"${n/1e9:.2f}B"
    if n>=1e6:  return f"${n/1e6:.2f}M"
    return f"${n:,.0f}"

def score_badge(s):
    if s>=4.5: return "strong","🟢 Excellent"
    if s>=3.5: return "strong","🟢 Strong"
    if s>=2.5: return "avg",  "🟡 Average"
    return "weak","🔴 Weak"

def render_chips(chips):
    html = ""
    for label,color in chips:
        css = f"chip-{color}"
        html += f'<span class="chip {css}">{label}</span>'
    return html

def render_data_integrity(meta, interval):
    if meta is None: return
    now = datetime.now()
    ft  = meta.get("fetch_time",now)
    lc  = meta.get("last_candle")
    bar_age = None
    if lc is not None:
        try:
            lc2 = pd.Timestamp(lc)
            if lc2.tzinfo: lc2 = lc2.tz_localize(None)
            bar_age = int((now-lc2).total_seconds()/60)
        except Exception: pass
    feed = "🔄 Near-Realtime" if bar_age and bar_age<20 else ("⏱ Delayed" if bar_age and bar_age<60 else "📦 Historical")
    mkt  = "🟢 Open" if (now.weekday()<5 and 9<=now.hour<16) else "🔴 Closed"
    st.markdown('<div class="data-panel">', unsafe_allow_html=True)
    d1,d2,d3,d4,d5,d6,d7 = st.columns(7)
    d1.metric("Source","Yahoo Finance"); d2.metric("Interval",interval)
    d3.metric("Fetched",ft.strftime("%H:%M:%S"))
    d4.metric("Last Bar",str(lc)[:16] if lc else "N/A")
    d5.metric("Bar Age",f"{bar_age}m" if bar_age else "N/A")
    d6.metric("Market",mkt); d7.metric("Feed",feed)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📊 Trading Desk")
    page = st.radio("nav", [
        "📈 Chart & Analysis","💼 Portfolio","👁️ Watchlist","📓 Journal",
    ], label_visibility="collapsed")
    st.divider()
    st.markdown("**🤖 AI Settings**")
    ai_provider = st.selectbox("AI Provider",["none","openrouter"],key="ai_provider")
    ai_model    = st.text_input("Model",value="openrouter/auto",key="ai_model")
    ai_enabled  = st.toggle("Enable AI Comments",value=True,key="ai_enabled")
    key_ok      = get_openrouter_key() is not None
    if ai_provider=="openrouter" and not key_ok:
        st.warning("⚠️ OPENROUTER_API_KEY missing")
    elif ai_provider=="openrouter" and key_ok:
        st.success("✅ API key loaded")
    st.divider()
    st.markdown("**⚡ Quick Charts**")
    for qt in ["AVGO","META","NVDA","AMD","AMZN"]:
        if st.button(qt,key=f"q_{qt}",use_container_width=True):
            st.session_state["ticker_input"]=qt
            st.session_state["page_override"]="📈 Chart & Analysis"
    st.divider()
    st.caption("Data: Yahoo Finance · 5min cache")

if "page_override" in st.session_state:
    page = st.session_state.pop("page_override")

# ══════════════════════════════════════════════════════════════
#  PAGE 1: CHART & ANALYSIS
# ══════════════════════════════════════════════════════════════
if "📈" in page:
    st.markdown("## 📈 Chart & Analysis")
    c1,c2,c3 = st.columns([3,1,1])
    with c1:
        default_tick = st.session_state.get("ticker_input","AVGO")
        ticker = st.text_input("Ticker",value=default_tick,placeholder="e.g. AVGO, META, NVDA").strip().upper()
    with c2:
        interval = st.selectbox("Interval",["1m","5m","15m","1h","4h","1d","1wk"],index=5)
    with c3:
        st.markdown("<br>",unsafe_allow_html=True)
        run = st.button("▶ Load",use_container_width=True)

    if not ticker: st.info("Enter a ticker symbol above."); st.stop()

    df,info,error,meta = get_stock_data(ticker, interval)
    if error or df is None:
        st.error(f"Could not load data for **{ticker}**: {error}"); st.stop()

    news              = get_news(ticker)
    sent_score,sent_label = calculate_sentiment(news)
    a                 = analyze(df, info or {}, ticker, sent_score)
    catalyst          = get_catalyst_data(ticker)
    opt_data          = get_options_data(ticker)
    mtf               = get_mtf_data(ticker)
    signals           = generate_signals(df, a, catalyst, mtf["alignment_score"], info or {})
    regime            = get_market_regime()

    cur  = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if len(df)>1 else cur
    chg  = cur - prev
    chg_pct = chg/prev*100 if prev else 0

    # ── Data integrity ──
    render_data_integrity(meta, interval)

    # ── Header metrics ──
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("Ticker",ticker)
    m2.metric("Price",f"${cur:.2f}",f"{chg:+.2f} ({chg_pct:+.2f}%)")
    m3.metric("RSI",f"{a['rsi']:.1f}","OB" if a['rsi']>70 else ("OS" if a['rsi']<30 else "✓"))
    m4.metric("Trend",a["trend_bias"])
    m5.metric("Score",f"{a['overall_score']:.1f}/5")
    m6.metric("Sentiment",f"{sent_score}/100 {sent_label}")
    st.divider()

    # ── Market Regime ──
    st.markdown("### 🌐 Market Regime")
    r1c,r2c,r3c,r4c = st.columns(4)
    r1c.metric("SPY",regime["SPY"]); r2c.metric("QQQ",regime["QQQ"])
    r3c.metric("VIX",f"{regime['VIX']:.1f}" if regime["VIX"] else "N/A",
               "⚠️ High" if regime["VIX"] and regime["VIX"]>25 else ("Normal" if regime["VIX"] else ""))
    reg_color = {"Risk-on":"#26a69a","Cautious Bull":"#f5c842","Risk-off":"#ef5350",
                 "Extreme Fear":"#ef5350","Choppy":"#f5c842"}.get(regime["regime"],"#787b86")
    r4c.markdown(f"**Regime:**<br><span style='color:{reg_color};font-size:18px;font-weight:700'>{regime['regime']}</span>",unsafe_allow_html=True)
    st.divider()

    # ── Signal Chips ──
    st.markdown("### ⚡ Signals & Warnings")
    st.markdown(render_chips(signals), unsafe_allow_html=True)
    st.divider()

    # ── Chart Controls ──
    st.markdown("**Chart Overlays**")
    oc1,oc2,oc3,oc4,oc5 = st.columns(5)
    with oc1: show_ema  = st.checkbox("EMAs",value=True)
    with oc2: show_bb   = st.checkbox("Bollinger",value=False)
    with oc3: show_vwap = st.checkbox("VWAP",value=True)
    with oc4: show_st   = st.checkbox("Supertrend",value=True)
    with oc5: show_macd = st.checkbox("MACD Panel",value=False)
    st.caption("💡 Scroll=zoom · Drag=pan · Non-trading gaps removed · Click range buttons above chart")
    fig = build_chart(df,ticker,interval,show_bb,show_vwap,show_st,show_ema,show_macd)
    st.plotly_chart(fig,use_container_width=True,config={
        "scrollZoom":True,"displayModeBar":True,
        "modeBarButtonsToAdd":["drawline","drawopenpath","drawrect","eraseshape"],
        "toImageButtonOptions":{"format":"png","filename":f"{ticker}_chart"},
    })
    st.divider()

    # ── Catalyst ──
    st.markdown("### 🗓️ Catalyst Panel")
    st.markdown(f'<div class="{catalyst["risk_css"]}"><strong>{catalyst["risk_label"]}</strong></div>',unsafe_allow_html=True)
    st.markdown("")
    ca1,ca2,ca3,ca4 = st.columns(4)
    with ca1:
        ed = catalyst["earnings_date"]
        st.metric("Next Earnings",str(ed) if ed else "N/A",
                  f"{catalyst['days_to_earnings']}d away" if catalyst["days_to_earnings"] is not None else "")
    ca2.metric("Event Risk",catalyst["event_risk"])
    ca3.metric("Ex-Div",str(catalyst["ex_div_date"]) if catalyst["ex_div_date"] else "N/A")
    at = catalyst["analyst_target"]
    if at:
        upside=(at-cur)/cur*100
        ca4.metric("Analyst Target",f"${at:.2f}",f"{upside:+.1f}%")
    else:
        ca4.metric("Analyst Target","N/A")
    st.divider()

    # ── Two-column: Trade Plan + Key Metrics ──
    st.markdown("### 📌 Trade Plan & Key Metrics")
    tp_col, km_col = st.columns(2)

    with tp_col:
        s1,s2,s3,r1,r2,r3 = a["levels"]
        pct_stop = (cur-a["stop_loss"])/cur*100 if cur>0 else 0
        rr_label = f"{a['rr']:.2f}:1"
        st.markdown('<div class="card card-accent-blue">', unsafe_allow_html=True)
        st.markdown("**Trade Plan**")
        st.markdown(f"""
<table class="trade-table">
<tr><td>Entry Zone</td><td style="color:#d1d4dc">${a['entry_low']:.2f} – ${a['entry_high']:.2f}</td></tr>
<tr><td>Stop Loss ({pct_stop:.1f}%)</td><td style="color:#ef5350">${a['stop_loss']:.2f}</td></tr>
<tr><td>Target 1 (+{((a['target_1']-cur)/cur*100):.1f}%)</td><td style="color:#26a69a">${a['target_1']:.2f}</td></tr>
<tr><td>Target 2 (+{((a['target_2']-cur)/cur*100):.1f}%)</td><td style="color:#26a69a">${a['target_2']:.2f}</td></tr>
<tr><td>Risk/Reward</td><td style="color:{'#26a69a' if a['rr']>=2 else '#f5c842'}">{rr_label}</td></tr>
<tr><td>Support S1</td><td>${s1:.2f}</td></tr>
<tr><td>Support S2</td><td>${s2:.2f}</td></tr>
<tr><td>Resistance R1</td><td>${r1:.2f}</td></tr>
<tr><td>Resistance R2</td><td>${r2:.2f}</td></tr>
</table>
""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with km_col:
        info_d = info or {}
        rev_g  = (info_d.get("revenueGrowth") or 0)*100
        eps_g  = (info_d.get("earningsGrowth") or 0)*100
        inst   = (info_d.get("heldPercentInstitutions") or 0)*100
        float_s= info_d.get("floatShares",0) or 0
        vol_cur= float(df["Volume"].iloc[-1])
        vol_avg= float(df["Vol_EMA20"].iloc[-1]) if "Vol_EMA20" in df.columns else vol_cur
        vr     = vol_cur/vol_avg if vol_avg>0 else 1.0

        def km_color(val, good, bad):
            if val>=good: return "#26a69a"
            if val<=bad:  return "#ef5350"
            return "#f5c842"

        st.markdown('<div class="card card-accent-gold">', unsafe_allow_html=True)
        st.markdown("**Key Metrics**")
        st.markdown(f"""
<table class="trade-table">
<tr><td>EPS Qtr Growth</td><td style="color:{km_color(eps_g,15,-5)}">{eps_g:.1f}%</td></tr>
<tr><td>Revenue Growth</td><td style="color:{km_color(rev_g,15,0)}">{rev_g:.1f}%</td></tr>
<tr><td>Gross Margin</td><td style="color:{km_color((info_d.get('grossMargins') or 0)*100,50,25)}">{(info_d.get('grossMargins') or 0)*100:.1f}%</td></tr>
<tr><td>Net Margin</td><td style="color:{km_color((info_d.get('profitMargins') or 0)*100,15,0)}">{(info_d.get('profitMargins') or 0)*100:.1f}%</td></tr>
<tr><td>Volume vs Avg</td><td style="color:{km_color(vr,1.3,0.7)}">{vr:.2f}x</td></tr>
<tr><td>Institutional %</td><td style="color:{km_color(inst,65,30)}">{inst:.1f}%</td></tr>
<tr><td>Float Shares</td><td>{fmt_large(float_s)}</td></tr>
<tr><td>P/E (TTM)</td><td>{f"{info_d.get('trailingPE',0):.1f}x" if info_d.get('trailingPE') else 'N/A'}</td></tr>
<tr><td>Fwd P/E</td><td>{f"{info_d.get('forwardPE',0):.1f}x" if info_d.get('forwardPE') else 'N/A'}</td></tr>
</table>
""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # ── MTF ──
    st.markdown("### 🕐 Multi-Timeframe Confirmation")
    align = mtf["alignment_score"]
    ac = "#26a69a" if align>=65 else ("#f5c842" if align>=45 else "#ef5350")
    st.markdown(f"**Alignment:** <span style='color:{ac};font-size:22px;font-weight:700'>{align}/100</span> — {mtf['primary_bias']}",unsafe_allow_html=True)
    ma,mb,mc,md = st.columns(4)
    ma.metric("Bias",mtf["primary_bias"])
    mb.metric("Trigger",mtf["trigger"][:38]+"…" if len(mtf["trigger"])>38 else mtf["trigger"])
    mc.metric("Conflict",mtf["conflict_note"][:38]+"…" if len(mtf["conflict_note"])>38 else mtf["conflict_note"])
    md.metric("Score",f"{align}/100")
    with st.expander("📋 Full Timeframe Table",expanded=False):
        st.dataframe(pd.DataFrame(mtf["rows"]),use_container_width=True,hide_index=True)
    st.divider()

    # ── Options ──
    st.markdown("### 🎯 Options Setup")
    opt_vehicle,opt_reason = options_recommendation(a["trend_bias"],catalyst["event_risk"],
                                                    opt_data.get("iv"),a["rsi"],a["atr_pct"])
    oa,ob,oc,od = st.columns(4)
    oa.metric("Recommendation",opt_vehicle)
    ob.metric("Next Expiry",opt_data.get("next_exp") or "N/A")
    oc.metric("IV (Median)",f"{opt_data['iv']:.1f}%" if opt_data.get("iv") else "N/A")
    od.metric("Event Risk",catalyst["event_risk"])
    st.info(f"**Reasoning:** {opt_reason}")
    if opt_data.get("error"):
        st.caption(f"ℹ️ {opt_data['error']}")
    st.divider()

    # ── AI Comments ──
    st.markdown("### 🤖 Expert AI Analysis")
    eff_prov = st.session_state.get("ai_provider","none")
    if not st.session_state.get("ai_enabled",True): eff_prov="none"
    if st.button("🤖 Generate Expert Analysis",key="gen_ai"):
        with st.spinner("Generating expert analysis..."):
            ai_payload = {
                "ticker":ticker,"price":a["price"],"trend_bias":a["trend_bias"],
                "rsi":a["rsi"],"alignment_score":mtf["alignment_score"],
                "primary_bias":mtf["primary_bias"],"event_risk":catalyst["event_risk"],
                "options_vehicle":opt_vehicle,"entry_low":a["entry_low"],"entry_high":a["entry_high"],
                "stop_loss":a["stop_loss"],"target_1":a["target_1"],"target_2":a["target_2"],
                "rr":a["rr"],"iv":opt_data.get("iv"),"sentiment_label":sent_label,
                "sentiment_score":sent_score,"atr_pct":a["atr_pct"],"overall_score":a["overall_score"],
                "support":a["support_watch"],"resistance":a["resistance_watch"],
                "fund_score":a["fund_score"],"tech_score":a["tech_score"],
                "rev_growth":a["rev_growth"],"eps_growth":a["eps_growth"],
                "inst_pct":a["inst_pct"],"max_dd":a["max_dd"],
                "ema9":a["ema9"],"ema20":a["ema20"],"ema50":a["ema50"],"ema200":a["ema200"],
                "vwap":a["vwap"],"days_to_earnings":catalyst["days_to_earnings"],
                "sector":a["sector"],
            }
            commentary,ai_warn = generate_ai_commentary(ai_payload,provider=eff_prov,
                                                         model=st.session_state.get("ai_model","openrouter/auto"))
        if ai_warn: st.warning(ai_warn)
        st.markdown('<div class="ai-box">', unsafe_allow_html=True)
        st.markdown(commentary)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.caption("Click the button above to generate comprehensive expert analysis.")
    st.divider()

    # ── News (scrollable, 20 items) ──
    nc1,nc2 = st.columns([1,2])
    with nc1:
        st.markdown("### 📊 Sentiment")
        sc = "#26a69a" if sent_label=="Bullish" else ("#ef5350" if sent_label=="Bearish" else "#f5c842")
        st.progress(sent_score/100)
        st.markdown(f"<span style='color:{sc};font-weight:700;font-size:16px'>{sent_label} — {sent_score}/100</span>",unsafe_allow_html=True)
        st.caption(f"Based on {len(news)} news items across multiple sources")

    with nc2:
        st.markdown(f"### 📰 Latest News ({len(news)} items)")
        st.markdown('<div class="news-scroll">', unsafe_allow_html=True)
        for item in news[:20]:
            pub   = item.get("published","")[:16] if item.get("published") else ""
            src   = item.get("source","")
            title = item["title"]
            # Inline sentiment
            t_low  = title.lower()
            bull   = sum(1 for w in BULL_WORDS if w in t_low)
            bear   = sum(1 for w in BEAR_WORDS if w in t_low)
            dot    = "🟢" if bull>bear else ("🔴" if bear>bull else "⚪")
            st.markdown(
                f'<div class="news-item">{dot} <a href="{item["link"]}" target="_blank" style="color:#7ab4ff;text-decoration:none;font-size:12px">{title}</a>'
                f'<br><span style="color:#787b86;font-size:10px">{src} · {pub}</span></div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)
    st.divider()

    # ── Risk Calculator ──
    st.markdown("### 💰 Position Sizer")
    with st.expander("Open Position Sizer",expanded=False):
        r1c,r2c = st.columns(2)
        with r1c:
            rc_acct  = st.number_input("Account Size ($)",value=50000.0,step=1000.0)
            rc_cash  = st.number_input("Cash Available ($)",value=float(CASH_USD),step=100.0)
            rc_riskp = st.number_input("Risk Per Trade (%)",value=1.0,min_value=0.1,max_value=5.0,step=0.1)
            rc_asset = st.selectbox("Asset Type",["Stock","Option"])
        with r2c:
            rc_entry = st.number_input("Entry ($)",value=float(f"{a['entry_high']:.2f}"))
            rc_stop  = st.number_input("Stop ($)", value=float(f"{a['stop_loss']:.2f}"))
            rc_t1    = st.number_input("Target 1 ($)",value=float(f"{a['target_1']:.2f}"))
            rc_t2    = st.number_input("Target 2 ($)",value=float(f"{a['target_2']:.2f}"))
        if st.button("Calculate",key="calc_pos"):
            res,warn = calc_position_size(rc_acct,rc_cash,rc_riskp,rc_entry,rc_stop,rc_t1,rc_t2,rc_asset,a["atr"])
            if res is None: st.error(warn)
            else:
                p1,p2,p3,p4,p5,p6 = st.columns(6)
                p1.metric("Max Risk",f"${res['max_risk']:,.2f}")
                p2.metric(f"{rc_asset}s",f"{res['units']} {res['label_unit']}")
                p3.metric("Position Value",f"${res['pos_value']:,.2f}")
                p4.metric("Risk/Unit",f"${res['risk_per_unit']:.2f}")
                p5.metric("R:R → T1",f"{res['rr1']:.2f}x")
                p6.metric("R:R → T2",f"{res['rr2']:.2f}x")
                if warn: st.warning(warn)
                st.session_state["last_pos_calc"]={"ticker":ticker,"entry":rc_entry,"stop":rc_stop,
                                                    "target1":rc_t1,"target2":rc_t2,"size":res["units"],"asset_type":rc_asset}
    st.divider()

    # ── 6-Pillar Summary ──
    st.markdown("### 📐 Analysis Scores")
    scores = a["scores"]
    cols6  = st.columns(6)
    labels = ["Fundamentals","Technicals","Risk","Plan","Entry/Exit","Mindset"]
    icons  = ["💼","📊","🛡️","📋","🎯","🧠"]
    for i,(label,icon) in enumerate(zip(labels,icons)):
        sc   = scores.get(label, scores.get(list(scores.keys())[i],3))
        _,badge = score_badge(sc)
        color   = "#26a69a" if sc>=3.5 else ("#f5c842" if sc>=2.5 else "#ef5350")
        cols6[i].markdown(
            f'<div class="card" style="text-align:center">'
            f'<div style="font-size:20px">{icon}</div>'
            f'<div style="color:#787b86;font-size:11px;margin:4px 0">{label}</div>'
            f'<div style="color:{color};font-size:22px;font-weight:700">{sc:.1f}</div>'
            f'</div>', unsafe_allow_html=True)

    overall_color = "#26a69a" if a["overall_score"]>=3.5 else ("#f5c842" if a["overall_score"]>=2.5 else "#ef5350")
    st.markdown(
        f'<div style="background:#1e222d;border:1px solid #2a2e3e;border-radius:8px;padding:12px 18px;margin-top:8px">'
        f'<span style="color:#787b86">Overall Score: </span>'
        f'<span style="color:{overall_color};font-size:20px;font-weight:700">{a["overall_score"]:.1f}/5</span>'
        f' &nbsp;|&nbsp; Trend: <strong>{a["trend_bias"]}</strong>'
        f' &nbsp;|&nbsp; Entry: <strong>${a["entry_low"]:.2f}–${a["entry_high"]:.2f}</strong>'
        f' &nbsp;|&nbsp; Stop: <strong style="color:#ef5350">${a["stop_loss"]:.2f}</strong>'
        f' &nbsp;|&nbsp; T1: <strong style="color:#26a69a">${a["target_1"]:.2f}</strong>'
        f' &nbsp;|&nbsp; R:R: <strong>{a["rr"]:.2f}x</strong>'
        f'</div>', unsafe_allow_html=True)
    st.divider()

    # ── Save to Journal ──
    with st.expander("💾 Save Setup to Journal",expanded=False):
        j1,j2 = st.columns(2)
        with j1:
            j_dir  = st.selectbox("Direction",["Long","Short"],key="j_dir")
            j_at   = st.selectbox("Asset Type",["Stock","Option","Spread"],key="j_at")
            j_th   = st.text_area("Thesis",key="j_th",placeholder="Why are you taking this trade?")
            j_tf   = st.selectbox("Timeframe",["1m","5m","15m","1h","4h","1d"],index=5,key="j_tf")
        with j2:
            j_res  = st.selectbox("Result",["Open","Win","Loss","Breakeven"],key="j_res")
            j_pnl  = st.number_input("P&L ($)",value=0.0,key="j_pnl")
            j_nt   = st.text_area("Notes",key="j_nt")
            j_mis  = st.text_area("Mistakes/Lessons",key="j_mis")
        lpc = st.session_state.get("last_pos_calc",{})
        if st.button("💾 Save",key="save_j"):
            save_trade({"date":datetime.now().strftime("%Y-%m-%d %H:%M"),"ticker":ticker,
                        "direction":j_dir,"asset_type":j_at,"entry":lpc.get("entry",a["entry_high"]),
                        "stop":lpc.get("stop",a["stop_loss"]),"target":lpc.get("target1",a["target_1"]),
                        "size":lpc.get("size",""),"thesis":j_th,"timeframe":j_tf,
                        "alignment_score":mtf["alignment_score"],"catalyst_state":catalyst["risk_label"],
                        "options_setup":opt_vehicle,"result":j_res,"pnl":j_pnl,"notes":j_nt,"mistakes":j_mis})
            st.success(f"✅ Saved: {ticker} {j_dir}")


# ══════════════════════════════════════════════════════════════
#  PAGE 2: PORTFOLIO
# ══════════════════════════════════════════════════════════════
elif "💼" in page:
    st.markdown("## 💼 Portfolio")

    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = DEFAULT_PORTFOLIO.copy()
    portfolio = st.session_state["portfolio"]

    # ── Fetch live prices ──
    total_invested = 0.0; total_value = 0.0; rows = []; alloc_labels = []; alloc_vals = []
    today_pnl = 0.0

    for tick, pos in portfolio.items():
        cur_price, prev_price = get_portfolio_price(tick)
        if cur_price is None: cur_price, prev_price = pos["avg_cost"], pos["avg_cost"]
        if prev_price is None: prev_price = cur_price
        invested   = pos["shares"] * pos["avg_cost"]
        value      = pos["shares"] * cur_price
        pnl        = value - invested
        pnl_pct    = pnl/invested*100 if invested else 0
        today_chg  = (cur_price - prev_price) * pos["shares"]
        today_pnl += today_chg
        to_target  = (pos["target"]-cur_price)/cur_price*100 if pos["target"] and cur_price else 0
        total_invested += invested; total_value += value
        # Stop signal: 7% below avg cost
        implied_stop = pos["avg_cost"] * 0.93
        at_risk      = cur_price < implied_stop
        rows.append({"Ticker":tick,"Shares":pos["shares"],"Avg Cost":f"${pos['avg_cost']:.4f}",
                     "Price":f"${cur_price:.2f}","Value":f"${value:,.2f}",
                     "P&L":f"${pnl:+,.2f} ({pnl_pct:+.2f}%)",
                     "Today":f"${today_chg:+,.2f}",
                     "Target":f"${pos['target']:.2f} (+{to_target:.1f}%)",
                     "Currency":pos["currency"],"⚠️":"🔴 NEAR STOP" if at_risk else "✅"})
        alloc_labels.append(tick); alloc_vals.append(value)

    total_pnl = total_value - total_invested
    total_pnl_pct = total_pnl/total_invested*100 if total_invested else 0
    cash_cad  = CASH_USD * CAD_RATE
    grand_total = total_value + cash_cad
    goal_mid  = (GOAL_LOW+GOAL_HIGH)/2

    # ── TOP SECTION: Portfolio Overview ──
    pv_col, td_col, at_col, gl_col = st.columns(4)
    pv_color = "#26a69a" if total_pnl>=0 else "#ef5350"
    td_color = "#26a69a" if today_pnl>=0 else "#ef5350"

    pv_col.markdown(f"""<div class="card card-accent-blue" style="text-align:center">
    <div style="color:#787b86;font-size:11px;text-transform:uppercase">Portfolio Value (CAD)</div>
    <div style="font-size:28px;font-weight:700;color:#d1d4dc">${total_value:,.2f}</div>
    <div style="color:#787b86;font-size:11px">+ ${cash_cad:,.0f} cash = <strong>${grand_total:,.0f}</strong></div>
    </div>""", unsafe_allow_html=True)

    td_col.markdown(f"""<div class="card" style="text-align:center">
    <div style="color:#787b86;font-size:11px;text-transform:uppercase">Today's Return</div>
    <div style="font-size:24px;font-weight:700;color:{td_color}">${today_pnl:+,.2f}</div>
    </div>""", unsafe_allow_html=True)

    at_col.markdown(f"""<div class="card" style="text-align:center">
    <div style="color:#787b86;font-size:11px;text-transform:uppercase">All-Time Return</div>
    <div style="font-size:24px;font-weight:700;color:{pv_color}">${total_pnl:+,.2f}</div>
    <div style="color:{pv_color};font-size:13px">{total_pnl_pct:+.2f}%</div>
    </div>""", unsafe_allow_html=True)

    goal_pct = min(grand_total/GOAL_LOW*100, 100) if GOAL_LOW>0 else 0
    gl_col.markdown(f"""<div class="card" style="text-align:center">
    <div style="color:#787b86;font-size:11px;text-transform:uppercase">Goal Progress</div>
    <div style="font-size:20px;font-weight:700;color:#2962ff">{goal_pct:.1f}%</div>
    <div style="color:#787b86;font-size:11px">${grand_total:,.0f} / ${GOAL_LOW:,}</div>
    </div>""", unsafe_allow_html=True)

    # Goal progress bar
    bar_w = min(100, goal_pct)
    st.markdown(f"""
    <div style="background:#1e222d;border-radius:6px;height:8px;margin:8px 0">
    <div style="background:linear-gradient(90deg,#2962ff,#26a69a);width:{bar_w}%;height:8px;border-radius:6px"></div>
    </div>
    <div style="text-align:center;color:#787b86;font-size:11px">Goal: ${GOAL_LOW:,}–${GOAL_HIGH:,} CAD</div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Charts + Holdings ──
    chart_col, hold_col = st.columns([1, 2])

    with chart_col:
        # Donut chart
        alloc_vals_with_cash = alloc_vals + [cash_cad]
        alloc_labels_with_cash = alloc_labels + ["Cash (CAD)"]
        colors = ["#2962ff","#26a69a","#f5c842","#ef5350","#b388ff","#ff9800","#787b86"]
        pie_fig = go.Figure(go.Pie(
            labels=alloc_labels_with_cash,
            values=alloc_vals_with_cash,
            hole=0.55,
            marker=dict(colors=colors[:len(alloc_vals_with_cash)],line=dict(color="#131722",width=2)),
            textinfo="label+percent",
            textfont=dict(size=11,color="#d1d4dc"),
        ))
        pie_fig.update_layout(
            template="plotly_dark",paper_bgcolor="#1e222d",plot_bgcolor="#1e222d",
            height=280,margin=dict(l=10,r=10,t=10,b=10),
            showlegend=False,
            annotations=[dict(text=f"${total_value:,.0f}",x=0.5,y=0.5,showarrow=False,
                              font=dict(size=13,color="#d1d4dc"),align="center")],
        )
        st.plotly_chart(pie_fig,use_container_width=True,config={"displayModeBar":False})

    with hold_col:
        st.markdown("**Holdings**")
        if rows:
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    # ── Stop-loss & Signal Alerts ──
    st.divider()
    st.markdown("### 🚨 Position Alerts & Signals")

    alert_shown = False
    for tick, pos in portfolio.items():
        cur_price, _ = get_portfolio_price(tick)
        if cur_price is None: cur_price = pos["avg_cost"]
        implied_stop = pos["avg_cost"] * 0.93   # 7% stop
        pnl_pct      = (cur_price-pos["avg_cost"])/pos["avg_cost"]*100
        target_pct   = (pos["target"]-cur_price)/cur_price*100 if pos["target"] else 0

        if cur_price < implied_stop:
            st.markdown(f'<div class="signal-alert">🔴 <strong>{tick} STOP LOSS ALERT</strong> — Price ${cur_price:.2f} has breached 7% stop (${implied_stop:.2f}). Review position immediately.</div>',unsafe_allow_html=True)
            alert_shown = True
        elif pnl_pct < -5:
            st.markdown(f'<div class="signal-alert">🟡 <strong>{tick} Warning</strong> — Down {pnl_pct:.1f}% from avg cost. Stop at ${implied_stop:.2f} ({7-abs(pnl_pct):.1f}% away).</div>',unsafe_allow_html=True)
            alert_shown = True
        elif pnl_pct > 15:
            st.markdown(f'<div class="signal-good">✅ <strong>{tick} Target Progress</strong> — Up {pnl_pct:.1f}%. {target_pct:.1f}% to target ${pos["target"]:.2f}. Consider partial profit at T1.</div>',unsafe_allow_html=True)
            alert_shown = True

    if not alert_shown:
        st.markdown('<div class="signal-good">✅ All positions within normal parameters. No stop-loss alerts.</div>',unsafe_allow_html=True)

    st.divider()

    # ── Add/Remove positions ──
    with st.expander("➕ Manage Positions"):
        pa,pb,pc,pd_c,pe = st.columns(5)
        with pa: ntick = st.text_input("Ticker (e.g. AVGO.TO)",key="ntick").strip().upper()
        with pb: nshar = st.number_input("Shares",min_value=0.0,key="nshar")
        with pc: navg  = st.number_input("Avg Cost",min_value=0.0,key="navg")
        with pd_c: ncur = st.selectbox("Currency",["CAD","USD"],key="ncur")
        with pe: ntgt  = st.number_input("Target",min_value=0.0,key="ntgt")
        if st.button("Save Position") and ntick:
            st.session_state["portfolio"][ntick]={"shares":nshar,"avg_cost":navg,"currency":ncur,"target":ntgt}
            st.success(f"Saved {ntick}"); st.rerun()
        rm = st.selectbox("Remove",["—"]+list(portfolio.keys()))
        if st.button("Remove") and rm!="—":
            del st.session_state["portfolio"][rm]; st.success(f"Removed {rm}"); st.rerun()

    st.divider()

    # ── Cash ──
    cx,cy = st.columns(2)
    cx.metric("USD Cash Reserve",f"${CASH_USD:,.2f}")
    cy.metric("CAD Equivalent (×1.38)",f"${cash_cad:,.0f}")

    st.divider()

    # ── AI STRATEGY SCANNER ──
    st.markdown("### 🧠 AI Strategy Scanner")
    st.caption("Scans your watchlist for the best setups and generates an AI portfolio strategy.")

    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = DEFAULT_WATCHLIST.copy()

    scan_tickers = st.session_state["watchlist"][:12]

    if st.button("🔍 Scan Watchlist & Generate Strategy",key="scan_strategy"):
        watchlist_signals = []
        with st.spinner(f"Scanning {len(scan_tickers)} watchlist stocks..."):
            progress = st.progress(0)
            for i,tick in enumerate(scan_tickers):
                progress.progress((i+1)/len(scan_tickers))
                df_w,info_w,err,_ = get_stock_data(tick,"1d")
                if err or df_w is None or len(df_w)<20: continue
                lat     = df_w.iloc[-1]
                price_w = float(lat["Close"])
                rsi_w   = float(lat["RSI"]) if "RSI" in df_w.columns else 50
                st_dir  = str(lat["SupertrendDir"]) if "SupertrendDir" in df_w.columns else "down"
                e20     = float(lat["EMA20"]) if "EMA20" in df_w.columns else price_w
                e50     = float(lat["EMA50"]) if "EMA50" in df_w.columns else price_w
                trend   = "Bullish" if price_w>e20>e50 else ("Bearish" if price_w<e20 else "Neutral")
                cat_w   = get_catalyst_data(tick)
                mtf_w   = get_mtf_data(tick)
                alg     = mtf_w["alignment_score"]
                # Signal quality
                if st_dir=="up" and alg>=60 and rsi_w<70 and price_w>e20:
                    sig = "🟢 BUY SETUP"
                elif st_dir=="down" and alg<=40:
                    sig = "🔴 AVOID"
                else:
                    sig = "🟡 WATCH"
                watchlist_signals.append({"ticker":tick,"trend":trend,"rsi":rsi_w,
                                           "st_dir":st_dir,"align":alg,"signal":sig,
                                           "price":price_w,"ema20":e20})
            progress.empty()

        # Sort by alignment
        watchlist_signals.sort(key=lambda x: x["align"], reverse=True)

        # Display top setups
        top3 = [s for s in watchlist_signals if s["signal"]=="🟢 BUY SETUP"][:4]
        if top3:
            st.markdown("#### 🟢 Top Watchlist Opportunities")
            for s in top3:
                pct_from_e20 = (s["price"]-s["ema20"])/s["ema20"]*100
                st.markdown(
                    f'<div class="signal-good">✅ <strong>{s["ticker"]}</strong> — '
                    f'${s["price"]:.2f} | MTF Align: {s["align"]}/100 | RSI: {s["rsi"]:.0f} | '
                    f'Trend: {s["trend"]} | ST: {s["st_dir"].upper()} | '
                    f'{pct_from_e20:+.1f}% from EMA20 → Consider entry on pullback to ${s["ema20"]:.2f}</div>',
                    unsafe_allow_html=True)

        # Generate AI strategy
        positions_with_price = {}
        for tick, pos in portfolio.items():
            cp, _ = get_portfolio_price(tick)
            if cp is None: cp = pos["avg_cost"]
            positions_with_price[tick] = {**pos, "current":cp,
                                           "pnl_pct":(cp-pos["avg_cost"])/pos["avg_cost"]*100}

        regime = get_market_regime()
        eff_prov = st.session_state.get("ai_provider","none")
        if not st.session_state.get("ai_enabled",True): eff_prov="none"

        with st.spinner("Generating portfolio strategy..."):
            strategy = generate_portfolio_strategy(positions_with_price, watchlist_signals, regime,
                                                    provider=eff_prov,
                                                    model=st.session_state.get("ai_model","openrouter/auto"))
        st.markdown("#### 📋 AI Portfolio Strategy")
        st.markdown('<div class="ai-box">', unsafe_allow_html=True)
        st.markdown(strategy)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="card" style="text-align:center;color:#787b86">'
                    '🔍 Click the button above to scan your watchlist and get an AI-powered portfolio strategy based on current market conditions.'
                    '</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE 3: WATCHLIST
# ══════════════════════════════════════════════════════════════
elif "👁️" in page:
    st.markdown("## 👁️ Watchlist Scanner")

    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = DEFAULT_WATCHLIST.copy()

    wa,wb = st.columns([3,1])
    with wa: add_t = st.text_input("Add ticker").strip().upper()
    with wb:
        st.markdown("<br>",unsafe_allow_html=True)
        if st.button("Add") and add_t and add_t not in st.session_state["watchlist"]:
            st.session_state["watchlist"].append(add_t); st.success(f"Added {add_t}"); st.rerun()
    rm_w = st.selectbox("Remove",["—"]+st.session_state["watchlist"])
    if st.button("Remove from watchlist") and rm_w!="—":
        st.session_state["watchlist"].remove(rm_w); st.rerun()

    st.divider()
    st.markdown("### 📡 Live Scan")
    st.caption("Click any ticker to open in Chart & Analysis.")

    wl_rows = []
    for tick in st.session_state["watchlist"]:
        df_w,info_w,_,_ = get_stock_data(tick,"1d")
        if df_w is None or df_w.empty: continue
        lat   = df_w.iloc[-1]
        prev  = df_w.iloc[-2] if len(df_w)>1 else lat
        pw    = float(lat["Close"])
        chg_w = (pw-float(prev["Close"]))/float(prev["Close"])*100
        rsi_w = float(lat["RSI"]) if "RSI" in df_w.columns else 0
        sd_w  = str(lat["SupertrendDir"]) if "SupertrendDir" in df_w.columns else "—"
        e20_w = float(lat["EMA20"]) if "EMA20" in df_w.columns else pw
        e50_w = float(lat["EMA50"]) if "EMA50" in df_w.columns else pw
        trend_w="▲ Bull" if pw>e20_w>e50_w else ("▼ Bear" if pw<e50_w else "→ Neutral")
        vol_w = float(lat["Volume"]); vol_avg_w = float(lat.get("Vol_EMA20",vol_w))
        vr_w  = vol_w/vol_avg_w if vol_avg_w>0 else 1.0
        # Multi-score for signal
        bull_pts = 0
        if pw>e20_w: bull_pts+=1
        if pw>e50_w: bull_pts+=1
        if sd_w=="up": bull_pts+=2
        if rsi_w>50 and rsi_w<70: bull_pts+=1
        if vr_w>1.2: bull_pts+=1
        sig = "🟢 BUY" if bull_pts>=5 else ("🟡 WATCH" if bull_pts>=3 else "🔴 AVOID")
        try:
            cat_w = get_catalyst_data(tick)
            if cat_w["days_to_earnings"] is not None and cat_w["days_to_earnings"]<=5:
                sig = "⚠️ EARNINGS"
        except Exception: pass
        wl_rows.append({"ticker":tick,"price":pw,"chg":chg_w,"rsi":rsi_w,"trend":trend_w,
                         "st":sd_w.upper(),"signal":sig,"vr":vr_w})

    if wl_rows:
        hdr = st.columns([1,1,1,1,1,1,1,1])
        for i,h in enumerate(["Ticker","Price","Chg%","RSI","Trend","ST","Vol/Avg","Signal"]):
            hdr[i].markdown(f"<span style='color:#787b86;font-size:11px;font-weight:700'>{h}</span>",unsafe_allow_html=True)
        for row in wl_rows:
            c1,c2,c3,c4,c5,c6,c7,c8 = st.columns([1,1,1,1,1,1,1,1])
            with c1:
                if st.button(row["ticker"],key=f"wl_{row['ticker']}"):
                    st.session_state["ticker_input"]=row["ticker"]
                    st.session_state["page_override"]="📈 Chart & Analysis"
                    st.rerun()
            c2.write(f"${row['price']:.2f}")
            chg_col = "#26a69a" if row["chg"]>=0 else "#ef5350"
            c3.markdown(f"<span style='color:{chg_col}'>{row['chg']:+.2f}%</span>",unsafe_allow_html=True)
            rsi_col = "#ef5350" if row["rsi"]>70 else ("#26a69a" if row["rsi"]<30 else "#d1d4dc")
            c4.markdown(f"<span style='color:{rsi_col}'>{row['rsi']:.0f}</span>",unsafe_allow_html=True)
            c5.write(row["trend"])
            st_col = "#26a69a" if row["st"]=="UP" else "#ef5350"
            c6.markdown(f"<span style='color:{st_col};font-weight:700'>{row['st']}</span>",unsafe_allow_html=True)
            vr_col = "#26a69a" if row["vr"]>1.3 else ("#ef5350" if row["vr"]<0.7 else "#d1d4dc")
            c7.markdown(f"<span style='color:{vr_col}'>{row['vr']:.2f}x</span>",unsafe_allow_html=True)
            c8.write(row["signal"])

# ══════════════════════════════════════════════════════════════
#  PAGE 4: JOURNAL
# ══════════════════════════════════════════════════════════════
elif "📓" in page:
    st.markdown("## 📓 Trade Journal & Performance")
    jdf = load_journal(); analytics = journal_analytics(jdf)

    if analytics:
        st.markdown("### 📊 Performance Summary")
        j1,j2,j3,j4 = st.columns(4)
        j1.metric("Total Trades",analytics["total"])
        wr_col = "#26a69a" if analytics["win_rate"]>=55 else ("#ef5350" if analytics["win_rate"]<40 else "#f5c842")
        j2.metric("Win Rate",f"{analytics['win_rate']:.1f}%")
        j3.metric("Total P&L",f"${analytics['total_pnl']:+,.2f}")
        j4.metric("Profit Factor",f"{analytics['profit_factor']:.2f}x" if analytics['profit_factor']!=float("inf") else "∞")
        jb,jc,jd,_ = st.columns(4)
        jb.metric("Avg Win",f"${analytics['avg_win']:+,.2f}")
        jc.metric("Avg Loss",f"${analytics['avg_loss']:+,.2f}")
        jd.metric("Expectancy",f"${analytics['expectancy']:+,.2f}")
        for ins in analytics.get("insights",[]): st.info(ins)

        if not jdf.empty and "pnl" in jdf.columns:
            pnl_s = pd.to_numeric(jdf["pnl"],errors="coerce").fillna(0)
            if pnl_s.sum()!=0:
                fig_p = go.Figure()
                fig_p.add_trace(go.Bar(x=list(range(1,len(pnl_s)+1)),y=pnl_s.tolist(),
                    marker_color=["#26a69a" if v>=0 else "#ef5350" for v in pnl_s],name="P&L"))
                fig_p.add_trace(go.Scatter(x=list(range(1,len(pnl_s)+1)),y=pnl_s.cumsum().tolist(),
                    mode="lines+markers",name="Cumulative",line=dict(color="#2962ff",width=2),yaxis="y2"))
                fig_p.update_layout(template="plotly_dark",plot_bgcolor="#131722",paper_bgcolor="#131722",
                    height=260,margin=dict(l=10,r=10,t=24,b=10),
                    yaxis2=dict(overlaying="y",side="right"),font=dict(color="#d1d4dc"))
                st.plotly_chart(fig_p,use_container_width=True)
        st.divider()

    st.markdown("### ➕ Log Trade")
    with st.expander("Open Form",expanded=not analytics):
        fa,fb = st.columns(2)
        with fa:
            ft    = st.text_input("Ticker",key="ft")
            fd    = st.selectbox("Direction",["Long","Short"],key="fd")
            fa_   = st.selectbox("Asset",["Stock","Option","Spread"],key="fa_")
            fe    = st.number_input("Entry ($)",value=0.0,key="fe")
            fs    = st.number_input("Stop ($)", value=0.0,key="fs")
            fg    = st.number_input("Target ($)",value=0.0,key="fg")
            fsz   = st.number_input("Size",value=0,key="fsz")
        with fb:
            fth   = st.text_area("Thesis",key="fth")
            ftf   = st.selectbox("Timeframe",["1m","5m","15m","1h","4h","1d"],index=5,key="ftf")
            faln  = st.slider("Alignment (0–100)",0,100,50,key="faln")
            fcat  = st.selectbox("Catalyst State",["Low Risk","Moderate Risk","High Risk"],key="fcat")
            fopt  = st.text_input("Options Setup",key="fopt")
            fres  = st.selectbox("Result",["Open","Win","Loss","Breakeven"],key="fres")
            fpnl  = st.number_input("P&L ($)",value=0.0,key="fpnl")
            fnt   = st.text_area("Notes",key="fnt")
            fmis  = st.text_area("Mistakes",key="fmis")
        if st.button("💾 Save Trade",key="save_mt"):
            if ft:
                save_trade({"date":datetime.now().strftime("%Y-%m-%d %H:%M"),"ticker":ft.upper(),
                            "direction":fd,"asset_type":fa_,"entry":fe,"stop":fs,"target":fg,
                            "size":fsz,"thesis":fth,"timeframe":ftf,"alignment_score":faln,
                            "catalyst_state":fcat,"options_setup":fopt,"result":fres,"pnl":fpnl,
                            "notes":fnt,"mistakes":fmis})
                st.success("Saved!"); st.rerun()

    st.divider()
    st.markdown("### 📋 Trade Log")
    if jdf.empty:
        st.info("No trades yet. Use the form above or save from the Chart page.")
    else:
        st.dataframe(jdf,use_container_width=True,hide_index=True)
        st.download_button("⬇ Download CSV",data=jdf.to_csv(index=False),
                           file_name="trade_journal.csv",mime="text/csv")
        if st.button("🗑 Delete Last Trade"):
            jdf = jdf.iloc[:-1]; jdf.to_csv(JOURNAL_FILE,index=False)
            st.success("Deleted."); st.rerun()
