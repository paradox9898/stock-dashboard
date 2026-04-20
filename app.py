"""
Utpal Professional Trading Dashboard v3
All analysis computed in Python. OpenRouter only for commentary.
API key loaded from st.secrets ONLY — never hardcoded.
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os, csv, requests, re

st.set_page_config(page_title="Utpal Trading Desk", layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
body,.stApp{background:#131722;color:#d1d4dc;font-family:'Inter',sans-serif;}
.stTextInput>div>input,.stSelectbox>div>div,.stNumberInput>div>input{
  background:#1e222d;color:#d1d4dc;border:1px solid #363a45;border-radius:6px;}
.stButton>button{background:#2962ff;color:#fff;border:none;border-radius:6px;
  padding:7px 18px;font-weight:600;letter-spacing:.3px;}
.stButton>button:hover{background:#1e53e5;}
.card{background:#1e222d;border:1px solid #2a2e3e;border-radius:10px;padding:16px 20px;margin-bottom:12px;}
.card-accent-green{border-left:4px solid #26a69a;}
.card-accent-red  {border-left:4px solid #ef5350;}
.card-accent-blue {border-left:4px solid #2962ff;}
.card-accent-gold {border-left:4px solid #f5c842;}
div[data-testid="stMetric"]{background:#1e222d;border:1px solid #2a2e3e;
  border-radius:8px;padding:10px 14px;}
div[data-testid="stMetricLabel"]{color:#787b86;font-size:11px;text-transform:uppercase;letter-spacing:.5px;}
div[data-testid="stMetricValue"]{color:#d1d4dc;font-size:20px;font-weight:700;}
[data-testid="stSidebar"]{background:#161b27;border-right:1px solid #2a2e3e;}
.risk-high{background:rgba(239,83,80,.15);border:1px solid #ef5350;border-radius:8px;padding:10px 16px;}
.risk-med {background:rgba(245,200,66,.10);border:1px solid #f5c842;border-radius:8px;padding:10px 16px;}
.risk-low {background:rgba(38,166,154,.10);border:1px solid #26a69a;border-radius:8px;padding:10px 16px;}
.chip{display:inline-block;padding:4px 10px;border-radius:20px;font-size:11px;
  font-weight:600;margin:3px 3px 3px 0;white-space:nowrap;}
.chip-green{background:rgba(38,166,154,.2);border:1px solid #26a69a;color:#26a69a;}
.chip-red  {background:rgba(239,83,80,.2);border:1px solid #ef5350;color:#ef5350;}
.chip-gold {background:rgba(245,200,66,.2);border:1px solid #f5c842;color:#f5c842;}
.chip-blue {background:rgba(41,98,255,.2);border:1px solid #2962ff;color:#7ab4ff;}
.ai-box{background:#161b27;border:1px solid #2962ff;border-radius:10px;
  padding:18px 22px;margin-top:10px;line-height:1.7;}
.news-scroll{max-height:340px;overflow-y:auto;padding-right:6px;}
.news-item{background:#1e222d;border:1px solid #2a2e3e;border-radius:8px;
  padding:10px 14px;margin-bottom:8px;}
.news-item:hover{border-color:#2962ff;}
.social-scroll{max-height:340px;overflow-y:auto;padding-right:6px;}
.social-item{background:#1e222d;border:1px solid #2a2e3e;border-radius:8px;
  padding:10px 14px;margin-bottom:8px;}
.trade-table{width:100%;border-collapse:collapse;}
.trade-table td{padding:6px 4px;border-bottom:1px solid #2a2e3e;font-size:13px;}
.trade-table td:first-child{color:#787b86;}
.trade-table td:last-child{text-align:right;font-weight:600;}
.signal-alert{background:rgba(239,83,80,.12);border:1px solid #ef5350;
  border-radius:8px;padding:10px 14px;margin-bottom:8px;}
.signal-good {background:rgba(38,166,154,.12);border:1px solid #26a69a;
  border-radius:8px;padding:10px 14px;margin-bottom:8px;}
.data-panel{background:#161b27;border:1px solid #2a2e3e;border-radius:8px;padding:12px 16px;margin-bottom:12px;}
/* Pillar expander polish */
.pillar-explain{background:#1a1f2e;border-left:3px solid #2962ff;border-radius:4px;
  padding:10px 14px;margin-top:8px;font-size:13px;line-height:1.6;}
/* Advanced tab cards */
.adv-card{background:#1e222d;border:1px solid #2a2e3e;border-radius:10px;padding:14px 18px;margin-bottom:10px;}
.adv-card h4{margin:0 0 10px 0;color:#d1d4dc;font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;}
.score-grade{font-size:40px;font-weight:800;line-height:1;}
.green{color:#26a69a;font-weight:700;}
.red{color:#ef5350;font-weight:700;}
.gold{color:#f5c842;font-weight:700;}
h1,h2,h3{color:#d1d4dc;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════
DEFAULT_PORTFOLIO = {
    "AVGO.TO": {"shares": 1005, "avg_cost": 13.4353, "currency": "CAD", "target": 20.0},
    "META.TO": {"shares": 417,  "avg_cost": 32.59,   "currency": "CAD", "target": 45.0},
}
CASH_USD  = 16147.0
CAD_RATE  = 1.38
GOAL_LOW  = 60000
GOAL_HIGH = 80000

DEFAULT_WATCHLIST = [
    "AVGO","META","NVDA","AMD","AMZN","MSFT","GOOGL",
    "DOCN","NET","COHR","MU","INTC","NFLX","CLS",
    "ALAB","VRT","DELL","CIEN","GLW","KEYS",
]
JOURNAL_FILE = "trade_journal.csv"
JOURNAL_COLS = ["date","ticker","direction","asset_type","entry","stop","target",
                "size","thesis","timeframe","alignment_score","catalyst_state",
                "options_setup","result","pnl","notes","mistakes"]

BULL_WORDS = {"beat","beats","surge","surges","strong","upgrade","upgrades","record","growth",
              "profit","soar","soars","buy","outperform","expands","positive","rebound","bullish",
              "momentum","raises","raised","tops","rally","rallies","breakout","accelerates",
              "boosts","exceed","exceeds","above","upside","outpaces","winning","strength"}
BEAR_WORDS = {"miss","misses","drop","drops","downgrade","downgrades","warning","weak","fall",
              "falls","lawsuit","loss","losses","decline","declines","probe","investigation",
              "negative","concern","concerns","bearish","cut","cuts","risks","below","selloff",
              "sell-off","layoffs","layoff","slows","slowdown","disappoints","disappointing",
              "warns","withdraws","reduces","faces","headwinds"}

INTERVAL_MAP = {
    "1m":"7d","5m":"30d","15m":"60d","1h":"730d",
    "4h":"730d","1d":"2y","1wk":"5y",
}

# ══════════════════════════════════════════════════════════════
#  OPENROUTER
# ══════════════════════════════════════════════════════════════
def _get_or_key():
    try: return st.secrets.get("OPENROUTER_API_KEY")
    except Exception: return None

def _call_openrouter(prompt, model, max_tokens=1200):
    api_key = _get_or_key()
    if not api_key: return None, "No API key"
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                     "HTTP-Referer": "https://utpal-trading-dashboard.streamlit.app",
                     "X-Title": "Utpal Trading"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},
            timeout=30)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"], None
        return None, f"HTTP {resp.status_code}"
    except Exception as e:
        return None, str(e)[:80]

# ══════════════════════════════════════════════════════════════
#  INDICATORS
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
    ml = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()
    sl = ml.ewm(span=signal, adjust=False).mean()
    return ml, sl, ml - sl

def atr(df, period=14):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift(1)).abs()
    lc = (df["Low"]  - df["Close"].shift(1)).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(com=period-1, adjust=False).mean()

def vwap_calc(df, interval="1d"):
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    tpv = tp * df["Volume"]
    if interval in ("1m","5m","15m","1h","4h"):
        idx = df.index
        try:
            dates = idx.tz_convert("America/New_York").normalize() if idx.tz else pd.DatetimeIndex(idx).normalize()
        except Exception:
            dates = pd.DatetimeIndex(idx).normalize()
        result = pd.Series(np.nan, index=idx)
        for d in dates.unique():
            mask = dates == d
            cum_tpv = tpv[mask].cumsum()
            cum_vol = df["Volume"][mask].cumsum().replace(0, np.nan)
            result[mask] = cum_tpv.values / cum_vol.values
        return result
    else:
        roll_tpv = tpv.rolling(20, min_periods=1).sum()
        roll_vol = df["Volume"].rolling(20, min_periods=1).sum().replace(0, np.nan)
        return roll_tpv / roll_vol

def bollinger_bands(series, period=20, std_dev=2):
    mid   = series.rolling(period).mean()
    sigma = series.rolling(period).std()
    return mid + std_dev*sigma, mid, mid - std_dev*sigma

def supertrend(df, period=10, multiplier=3):
    data = df.copy()
    hl2  = (data["High"] + data["Low"]) / 2
    data["ATR14"] = atr(data, period)
    ub = hl2 + multiplier * data["ATR14"]
    lb = hl2 - multiplier * data["ATR14"]
    fub, flb = ub.copy(), lb.copy()
    for i in range(1, len(data)):
        fub.iloc[i] = (min(ub.iloc[i], fub.iloc[i-1])
                       if data["Close"].iloc[i-1] <= fub.iloc[i-1] else ub.iloc[i])
        flb.iloc[i] = (max(lb.iloc[i], flb.iloc[i-1])
                       if data["Close"].iloc[i-1] >= flb.iloc[i-1] else lb.iloc[i])
    trend = pd.Series(index=data.index, dtype="float64")
    direction = pd.Series(index=data.index, dtype="object")
    for i in range(len(data)):
        if i == 0:
            trend.iloc[0] = flb.iloc[0]; direction.iloc[0] = "up"; continue
        if trend.iloc[i-1] == fub.iloc[i-1]:
            if data["Close"].iloc[i] <= fub.iloc[i]:
                trend.iloc[i] = fub.iloc[i]; direction.iloc[i] = "down"
            else:
                trend.iloc[i] = flb.iloc[i]; direction.iloc[i] = "up"
        else:
            if data["Close"].iloc[i] >= flb.iloc[i]:
                trend.iloc[i] = flb.iloc[i]; direction.iloc[i] = "up"
            else:
                trend.iloc[i] = fub.iloc[i]; direction.iloc[i] = "down"
    return trend, direction

def get_swing_levels(df, lookback=5):
    highs, lows = [], []
    h, l = df["High"].values, df["Low"].values
    n = len(df)
    for i in range(lookback, n - lookback):
        if h[i] == max(h[i-lookback:i+lookback+1]): highs.append(float(h[i]))
        if l[i] == min(l[i-lookback:i+lookback+1]): lows.append(float(l[i]))
    def dedup(levels, reverse=False):
        seen, out = [], []
        for p in sorted(levels, reverse=reverse):
            if not any(abs(p-s)/max(s,0.01)<0.01 for s in seen):
                seen.append(p); out.append(p)
        return out[:6]
    return dedup(highs, reverse=True), dedup(lows)

def get_key_levels(df):
    price = float(df["Close"].iloc[-1])
    sh, sl = get_swing_levels(df)
    supports    = sorted([x for x in sl if x < price*1.005], reverse=True)
    resistances = sorted([x for x in sh if x > price*0.995])
    fb_s = [float(df["Low"].tail(n).min()) for n in [20,50,200] if len(df)>=n]
    fb_r = [float(df["High"].tail(n).max()) for n in [20,50,200] if len(df)>=n]
    while len(supports)    < 3: supports.append(fb_s[min(len(supports),len(fb_s)-1)] if fb_s else price*0.95)
    while len(resistances) < 3: resistances.append(fb_r[min(len(resistances),len(fb_r)-1)] if fb_r else price*1.05)
    return supports[0],supports[1],supports[2], resistances[0],resistances[1],resistances[2]

# ══════════════════════════════════════════════════════════════
#  DATA FETCH
# ══════════════════════════════════════════════════════════════
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
            return float(df["Close"].iloc[-1]), prev
    except Exception: pass
    return None, None

@st.cache_data(ttl=120)
def get_live_price(ticker):
    """Fast price fetch for regime panel live prices."""
    try:
        df = yf.Ticker(ticker).history(period="2d", interval="1d", auto_adjust=False)
        if df is not None and not df.empty:
            cur  = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2]) if len(df)>1 else cur
            return cur, (cur-prev)/prev*100
    except Exception: pass
    return None, None

# ══════════════════════════════════════════════════════════════
#  MARKET REGIME
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=600)
def get_market_regime():
    result = {"SPY":"Unknown","QQQ":"Unknown","VIX":None,"regime":"Unknown",
              "SPY_price":None,"SPY_chg":None,"QQQ_price":None,"QQQ_chg":None}
    for sym in ["SPY","QQQ"]:
        try:
            df = yf.Ticker(sym).history(period="60d",interval="1d",auto_adjust=False)
            if df is not None and not df.empty:
                p   = float(df["Close"].iloc[-1])
                prv = float(df["Close"].iloc[-2]) if len(df)>1 else p
                e20 = float(df["Close"].ewm(span=20,adjust=False).mean().iloc[-1])
                e50 = float(df["Close"].ewm(span=50,adjust=False).mean().iloc[-1])
                chg = (p-prv)/prv*100
                result[f"{sym}_price"] = p
                result[f"{sym}_chg"]   = chg
                result[sym] = "Bullish" if (p>e20 and p>e50) else ("Bearish" if (p<e20 and p<e50) else "Chop")
        except Exception: pass
    try:
        vix = yf.Ticker("^VIX").history(period="5d",interval="1d",auto_adjust=False)
        if vix is not None and not vix.empty:
            result["VIX"] = float(vix["Close"].iloc[-1])
    except Exception: pass
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
#  NEWS + SENTIMENT + SOCIAL
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=900)
def get_news(ticker):
    base  = ticker.split(".")[0]
    items = []
    queries = [
        f"{base}+stock", f"{base}+earnings+revenue",
        f"{base}+analyst+price+target",
        f"{base}+semiconductor" if base in ["AVGO","AMD","NVDA","MU","INTC","COHR"] else f"{base}+AI",
        f"{base}+outlook+guidance",
    ]
    seen = set()
    for q in queries:
        try:
            feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en")
            for e in feed.entries[:8]:
                title = e.title[:140]
                if title not in seen:
                    seen.add(title)
                    items.append({"title":title,"link":e.link,
                                  "published":getattr(e,"published","")[:25],
                                  "source":getattr(e,"source",{}).get("title","Google News") if hasattr(e,"source") else "Google News"})
        except Exception: pass
    try:
        yf_feed = feedparser.parse(f"https://finance.yahoo.com/rss/headline?s={base}")
        for e in yf_feed.entries[:8]:
            title = e.title[:140]
            if title not in seen:
                seen.add(title)
                items.append({"title":title,"link":e.link,"published":getattr(e,"published","")[:25],"source":"Yahoo Finance"})
    except Exception: pass
    return items[:30]

def calculate_sentiment(news_items):
    score, hits = 0, 0
    for item in news_items:
        t    = item["title"].lower()
        bull = sum(1 for w in BULL_WORDS if w in t)
        bear = sum(1 for w in BEAR_WORDS if w in t)
        wt   = 1.5 if any(x in t for x in ["analyst","target","upgrade","downgrade","earnings"]) else 1.0
        score += (bull-bear)*wt; hits += (bull+bear)*wt
    norm  = 50 if hits==0 else int(max(0, min(100, 50+max(-42, min(42, score*7)))))
    label = "Bullish" if norm>=63 else ("Bearish" if norm<=37 else "Neutral")
    return norm, label

@st.cache_data(ttl=1800)
def get_social_comments(ticker):
    """Fetch social-style commentary from Reddit/StockTwits via RSS proxies."""
    base  = ticker.split(".")[0]
    items = []
    seen  = set()
    # Reddit finance RSS
    reddit_subs = ["wallstreetbets","stocks","investing","options","StockMarket"]
    for sub in reddit_subs[:3]:
        try:
            url  = f"https://www.reddit.com/r/{sub}/search.rss?q={base}&sort=new&limit=10"
            feed = feedparser.parse(url)
            for e in feed.entries[:6]:
                title = (e.title or "")[:140].strip()
                body  = re.sub(r'<[^>]+>','',getattr(e,'summary','')).strip()[:160]
                if title and title not in seen:
                    seen.add(title)
                    bull = sum(1 for w in BULL_WORDS if w in title.lower()+body.lower())
                    bear = sum(1 for w in BEAR_WORDS if w in title.lower()+body.lower())
                    sentiment = "🟢" if bull>bear else ("🔴" if bear>bull else "⚪")
                    items.append({"platform":f"r/{sub}","text":title,
                                  "body":body,"sentiment":sentiment,
                                  "link":getattr(e,"link","#"),
                                  "date":getattr(e,"published","")[:16]})
        except Exception: pass
    # StockTwits-style (use Google News finance community signals)
    try:
        q    = f"{base}+stock+buy+sell+bullish+bearish"
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en")
        for e in feed.entries[:8]:
            title = (e.title or "")[:140].strip()
            if title and title not in seen:
                seen.add(title)
                bull = sum(1 for w in BULL_WORDS if w in title.lower())
                bear = sum(1 for w in BEAR_WORDS if w in title.lower())
                sentiment = "🟢" if bull>bear else ("🔴" if bear>bull else "⚪")
                items.append({"platform":"Market Feed","text":title,"body":"",
                              "sentiment":sentiment,"link":e.link,
                              "date":getattr(e,"published","")[:16]})
    except Exception: pass
    return items[:30]

# ══════════════════════════════════════════════════════════════
#  INSIDER ACTIVITY
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def get_insider_activity(ticker):
    """Fetch insider transactions from yfinance and summarize."""
    base = ticker.split(".")[0]
    result = {"transactions": [], "summary": "", "net_signal": "Neutral"}
    try:
        stk = yf.Ticker(base)
        ins = stk.insider_transactions
        if ins is not None and not ins.empty:
            ins = ins.head(15).copy()
            # Normalize columns
            cols = {c.lower().replace(" ","_"): c for c in ins.columns}
            buy_val = sell_val = 0
            rows = []
            for _, row in ins.iterrows():
                try:
                    insider = str(row.get("Insider","") or row.get("Name","") or "Unknown")[:30]
                    shares  = int(row.get("Shares",0) or 0)
                    value   = float(row.get("Value",0) or 0)
                    tx_type = str(row.get("Transaction","") or row.get("Type","") or "")
                    date    = str(row.get("Start Date","") or row.get("Date","") or "")[:10]
                    is_buy  = any(x in tx_type.lower() for x in ["buy","purchase","acquisition"])
                    is_sell = any(x in tx_type.lower() for x in ["sell","sale","disposition"])
                    if is_buy:  buy_val  += abs(value)
                    if is_sell: sell_val += abs(value)
                    rows.append({"Insider":insider,"Shares":f"{shares:,}","Value":f"${abs(value):,.0f}",
                                 "Type":tx_type[:20],"Date":date,"Signal":"🟢 Buy" if is_buy else ("🔴 Sell" if is_sell else "⚪ Other")})
                except Exception: pass
            result["transactions"] = rows[:10]
            # Summary
            if buy_val > sell_val * 2:
                result["net_signal"] = "Bullish"
                result["summary"] = f"Insiders net buyers — ${buy_val/1e6:.1f}M purchased vs ${sell_val/1e6:.1f}M sold in recent filings. Insider buying often signals confidence in near-term outlook."
            elif sell_val > buy_val * 2:
                result["net_signal"] = "Bearish"
                result["summary"] = f"Insiders net sellers — ${sell_val/1e6:.1f}M sold vs ${buy_val/1e6:.1f}M purchased. Note: insider selling may reflect diversification, not necessarily negative view."
            else:
                result["net_signal"] = "Neutral"
                result["summary"] = f"Mixed insider activity — ${buy_val/1e6:.1f}M bought, ${sell_val/1e6:.1f}M sold. No strong directional signal from insider transactions."
    except Exception as e:
        result["summary"] = f"Insider data unavailable: {str(e)[:60]}"
    return result

# ══════════════════════════════════════════════════════════════
#  CATALYST
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def get_catalyst_data(ticker):
    result = {"earnings_date":None,"days_to_earnings":None,"ex_div_date":None,
              "analyst_target":None,"event_risk":"Low","risk_label":"🟢 Low Risk","risk_css":"risk-low"}
    try:
        stock = yf.Ticker(ticker); cal = stock.calendar; info = stock.info or {}
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
            except Exception: pass
        try:
            ex = info.get("exDividendDate")
            if ex: result["ex_div_date"] = pd.Timestamp(ex,unit="s").date()
        except Exception: pass
        result["analyst_target"] = info.get("targetMeanPrice")
    except Exception: pass
    return result

# ══════════════════════════════════════════════════════════════
#  OPTIONS
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=1800)
def get_options_data(ticker):
    result = {"iv":None,"next_exp":None,"has_options":False,"error":None}
    try:
        stock = yf.Ticker(ticker); exps = stock.options
        if not exps: result["error"]="No options chain"; return result
        result.update({"has_options":True,"next_exp":exps[0]})
        calls = stock.option_chain(exps[0]).calls
        if "impliedVolatility" in calls.columns and not calls.empty:
            result["iv"] = float(calls["impliedVolatility"].dropna().median())*100
    except Exception as e:
        result["error"] = str(e)
    return result

def options_recommendation(trend, event_risk, iv, rsi_v, atr_pct):
    if event_risk=="High": return "⚠️ Avoid / Wait","Earnings ≤5 days — wait post-event."
    if iv is None:
        if trend=="Bullish" and rsi_v<65: return "📈 Stock / Calls","No IV. Bullish trend — stock or ATM calls."
        if trend=="Bearish": return "📉 Puts / Short","No IV. Bearish — puts or short."
        return "⏳ Wait","Trend unclear."
    if iv>60:
        if trend=="Bullish": return "🐂 Bull Call Spread",f"IV={iv:.1f}% high — spread limits debit."
        if trend=="Bearish": return "🐻 Bear Put Spread",f"IV={iv:.1f}% high — spread keeps cost down."
        return "⏳ Wait",f"IV={iv:.1f}% but no clear direction."
    if iv<=30:
        if trend=="Bullish" and rsi_v<65: return "📈 Calls (ATM)",f"Low IV={iv:.1f}% — cheap directional."
        if trend=="Bearish": return "📉 Puts (ATM)",f"Low IV={iv:.1f}% — cheap puts."
    if trend=="Bullish" and rsi_v<65: return "📈 Calls",f"Moderate IV={iv:.1f}%, bullish."
    if trend=="Bearish": return "📉 Puts",f"Moderate IV={iv:.1f}%, bearish."
    if atr_pct>4: return "🔄 Stock Only",f"High ATR={atr_pct:.1f}% — options expensive."
    return "⏳ Wait","Setup not clear enough."

# ══════════════════════════════════════════════════════════════
#  SCORING ENGINE
# ══════════════════════════════════════════════════════════════
def score_fundamentals(info):
    pts = 0.0; reasons = {"positive": [], "negative": []}
    rev_g=info.get("revenueGrowth"); gm=info.get("grossMargins"); nm=info.get("profitMargins")
    pe=info.get("trailingPE"); fwd_pe=info.get("forwardPE"); peg=info.get("pegRatio")
    debt=info.get("totalDebt",0) or 0; cash=info.get("totalCash",0) or 0
    fcf=info.get("freeCashflow",0) or 0; roe=info.get("returnOnEquity",0) or 0
    eps_g=info.get("earningsGrowth",0) or 0
    if rev_g is not None:
        if rev_g>0.25:   pts+=1.0; reasons["positive"].append(f"Strong revenue growth {rev_g*100:.1f}% (>25%)")
        elif rev_g>0.15: pts+=0.75; reasons["positive"].append(f"Good revenue growth {rev_g*100:.1f}%")
        elif rev_g>0.05: pts+=0.5; reasons["positive"].append(f"Moderate revenue growth {rev_g*100:.1f}%")
        elif rev_g>0:    pts+=0.25
        else:            pts-=0.5; reasons["negative"].append(f"Revenue declining {rev_g*100:.1f}%")
    if gm is not None:
        if gm>0.65:   pts+=1.0; reasons["positive"].append(f"Excellent gross margin {gm*100:.1f}% (>65%)")
        elif gm>0.50: pts+=0.75; reasons["positive"].append(f"Strong gross margin {gm*100:.1f}%")
        elif gm>0.35: pts+=0.5
        elif gm>0.20: pts+=0.25
        else:         pts-=0.25; reasons["negative"].append(f"Low gross margin {gm*100:.1f}%")
    if nm is not None:
        if nm>0.25:   pts+=0.75; reasons["positive"].append(f"Excellent net margin {nm*100:.1f}%")
        elif nm>0.15: pts+=0.5; reasons["positive"].append(f"Good net margin {nm*100:.1f}%")
        elif nm>0.05: pts+=0.25
        elif nm<0:    pts-=0.75; reasons["negative"].append(f"Negative net margin {nm*100:.1f}%")
    if eps_g>0.20:   pts+=0.5; reasons["positive"].append(f"Strong EPS growth {eps_g*100:.1f}%")
    elif eps_g>0.10: pts+=0.25
    elif eps_g<-0.10: pts-=0.25; reasons["negative"].append(f"EPS declining {eps_g*100:.1f}%")
    if fcf>0: pts+=0.5; reasons["positive"].append(f"Positive FCF ${fcf/1e9:.2f}B")
    elif fcf<0: pts-=0.25; reasons["negative"].append("Negative free cash flow")
    net_cash = cash - debt
    if net_cash>0: pts+=0.5; reasons["positive"].append(f"Net cash position ${net_cash/1e9:.2f}B")
    else: pts-=0.25; reasons["negative"].append(f"Net debt ${abs(net_cash)/1e9:.2f}B")
    if pe is not None and pe>0:
        if pe<18:   pts+=0.5; reasons["positive"].append(f"Attractive P/E {pe:.1f}x")
        elif pe<30: pts+=0.25
        elif pe>70: pts-=0.5; reasons["negative"].append(f"Very high P/E {pe:.1f}x — priced for perfection")
    if roe>0.25: pts+=0.5; reasons["positive"].append(f"Excellent ROE {roe*100:.1f}%")
    elif roe>0.15: pts+=0.25
    elif roe<0: pts-=0.25; reasons["negative"].append(f"Negative ROE {roe*100:.1f}%")
    return max(1.0, min(5.0, round(pts,1))), reasons

def score_technicals(df):
    if df is None or len(df)<20: return 3.0,"Neutral",{},{"positive":[],"negative":[]}
    lat=df.iloc[-1]; price=float(lat["Close"])
    e9=float(lat["EMA9"]); e20=float(lat["EMA20"]); e50=float(lat["EMA50"]); e200=float(lat["EMA200"])
    rsi_v=float(lat["RSI"]); macd_v=float(lat["MACD"]); macd_s=float(lat["MACD_Signal"])
    st_dir=str(lat["SupertrendDir"]); vwap_v=float(lat["VWAP"])
    vol=float(lat["Volume"]); vol_avg=float(lat.get("Vol_EMA20",vol))
    pts=0.0; reasons={"positive":[],"negative":[]}
    if price>e200: pts+=1.0; reasons["positive"].append(f"Price ${price:.2f} above EMA200 (${e200:.2f}) — long-term uptrend")
    else:          pts-=0.5; reasons["negative"].append(f"Price below EMA200 (${e200:.2f}) — long-term downtrend")
    if price>e50:  pts+=0.75; reasons["positive"].append(f"Price above EMA50 (${e50:.2f}) — medium-term strength")
    else:          reasons["negative"].append(f"Price below EMA50 (${e50:.2f})")
    if price>e20:  pts+=0.5; reasons["positive"].append(f"Price above EMA20 (${e20:.2f}) — short-term momentum")
    else:          reasons["negative"].append(f"Price below EMA20 (${e20:.2f})")
    if price>e9:   pts+=0.25; reasons["positive"].append("Price above EMA9 — immediate momentum bullish")
    if st_dir=="up": pts+=0.75; reasons["positive"].append("Supertrend bullish — trend following confirmed")
    else:            pts-=0.25; reasons["negative"].append("Supertrend bearish — trend following negative")
    if macd_v>macd_s: pts+=0.5; reasons["positive"].append(f"MACD {macd_v:.3f} above signal {macd_s:.3f} — bullish momentum")
    else:             reasons["negative"].append(f"MACD below signal — bearish momentum")
    if 45<rsi_v<65:   pts+=0.5; reasons["positive"].append(f"RSI {rsi_v:.1f} in healthy bull zone (45–65)")
    elif 65<=rsi_v<75: pts+=0.15; reasons["positive"].append(f"RSI {rsi_v:.1f} extended but not extreme")
    elif rsi_v>=75:    pts-=0.25; reasons["negative"].append(f"RSI {rsi_v:.1f} overbought — pullback risk")
    elif rsi_v<30:     pts-=0.5;  reasons["negative"].append(f"RSI {rsi_v:.1f} oversold — downtrend pressure")
    if price>vwap_v: pts+=0.25; reasons["positive"].append(f"Price above VWAP (${vwap_v:.2f}) — intraday buyers in control")
    else:            reasons["negative"].append(f"Price below VWAP (${vwap_v:.2f}) — sellers in control")
    if vol_avg>0 and vol>vol_avg*1.2: pts+=0.25; reasons["positive"].append(f"Volume {vol/vol_avg:.1f}x avg — strong participation")
    if e9>e20>e50>e200 and price>e9: pts+=0.5; reasons["positive"].append("Perfect EMA bull stack: EMA9>EMA20>EMA50>EMA200")
    trend = "Bullish" if pts>=2.5 else ("Bearish" if pts<1.2 else "Neutral")
    return max(1.0,min(5.0,round(pts,1))), trend, {
        "price":price,"ema9":e9,"ema20":e20,"ema50":e50,"ema200":e200,
        "rsi":rsi_v,"macd":macd_v,"macd_sig":macd_s,"st_dir":st_dir,"vwap":vwap_v
    }, reasons

def score_risk(df, atr_v, price):
    reasons = {"positive":[],"negative":[]}
    daily_ret = df["Close"].pct_change().dropna()
    vol_daily = float(daily_ret.std())*100
    max_dd    = float(((df["Close"]/df["Close"].cummax())-1).min())*100
    if vol_daily<1.0:   rs=4.5; reasons["positive"].append(f"Low daily volatility {vol_daily:.2f}% — smooth trend likely")
    elif vol_daily<2.0: rs=3.5; reasons["positive"].append(f"Moderate volatility {vol_daily:.2f}%")
    elif vol_daily<3.0: rs=3.0
    elif vol_daily<4.0: rs=2.5; reasons["negative"].append(f"High volatility {vol_daily:.2f}% — widen stops")
    else:               rs=2.0; reasons["negative"].append(f"Very high volatility {vol_daily:.2f}% — reduce size")
    if max_dd<-40: rs=max(1.5,rs-0.5); reasons["negative"].append(f"Max drawdown {max_dd:.1f}% — historically volatile")
    elif max_dd<-20: reasons["negative"].append(f"Max drawdown {max_dd:.1f}%")
    else: reasons["positive"].append(f"Max drawdown {max_dd:.1f}% — relatively contained")
    atr_pct = atr_v/price*100
    if atr_pct<1.5: reasons["positive"].append(f"ATR {atr_pct:.1f}% — tight stops possible")
    elif atr_pct>4: reasons["negative"].append(f"ATR {atr_pct:.1f}% — wide daily range, options expensive")
    return max(1.5,min(5.0,round(rs,1))), reasons, vol_daily, max_dd

def score_plan(trend, sentiment_score, ema20, ema50):
    reasons = {"positive":[],"negative":[]}
    if trend=="Bullish" and sentiment_score>=55:
        ps=4.0
        reasons["positive"].append(f"Bullish trend aligned with positive sentiment ({sentiment_score}/100)")
        reasons["positive"].append(f"Buy pullbacks to EMA20 (${ema20:.2f}) — high probability zone")
    elif trend=="Bullish":
        ps=3.5; reasons["positive"].append("Bullish trend but neutral sentiment — wait for pullbacks")
        reasons["negative"].append(f"Sentiment only {sentiment_score}/100 — less conviction")
    elif trend=="Bearish" and sentiment_score<40:
        ps=1.5; reasons["negative"].append("Bearish trend + negative sentiment — double confirmation of downtrend")
    elif trend=="Bearish":
        ps=2.0; reasons["negative"].append("Bearish trend — avoid longs until EMA50 reclaimed")
    else:
        ps=2.5; reasons["positive"].append("Mixed — wait for directional confirmation")
        reasons["negative"].append("No clear plan without trend direction")
    return ps, reasons

def score_entry_exit(trend, rr, entry_high, stop_loss, target_1, target_2, price):
    reasons = {"positive":[],"negative":[]}
    if trend=="Bullish" and rr>=2.0:
        es=4.0; reasons["positive"].append(f"R:R = {rr:.1f}x — excellent risk/reward")
    elif rr>=1.5:
        es=3.0; reasons["positive"].append(f"R:R = {rr:.1f}x — acceptable but could be better")
    else:
        es=2.0; reasons["negative"].append(f"R:R = {rr:.1f}x — below 1.5x minimum, improve entry or skip")
    pct_stop = (price-stop_loss)/price*100
    if pct_stop<3: reasons["positive"].append(f"Tight stop {pct_stop:.1f}% — low capital at risk")
    elif pct_stop>10: reasons["negative"].append(f"Wide stop {pct_stop:.1f}% — requires small position size")
    reasons["positive"].append(f"T1: ${target_1:.2f} (+{((target_1-price)/price*100):.1f}%), T2: ${target_2:.2f} (+{((target_2-price)/price*100):.1f}%)")
    return es, reasons

def score_mindset(vol_daily, rsi_v, near_resistance):
    reasons = {"positive":[],"negative":[]}
    if vol_daily>3.5:   ms=2.5; reasons["negative"].append("High volatility amplifies emotional mistakes — size down")
    elif vol_daily>2.0: ms=3.5
    else:               ms=4.5; reasons["positive"].append("Low volatility — calmer price action, easier to manage")
    if rsi_v>70: reasons["negative"].append(f"RSI {rsi_v:.0f} overbought — FOMO risk high, avoid chasing")
    if rsi_v<30: reasons["positive"].append(f"RSI {rsi_v:.0f} oversold — fear is high, but bounces are sharp")
    if near_resistance: reasons["negative"].append("Price near resistance — wait for breakout confirmation before adding")
    reasons["positive"].append("Always define stop BEFORE entry. Write it down. Honor it.")
    return ms, reasons

# ══════════════════════════════════════════════════════════════
#  FULL ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════
def analyze(df, info, ticker, sentiment_score):
    lat=df.iloc[-1]; price=float(lat["Close"])
    e9=float(lat["EMA9"]); e20=float(lat["EMA20"]); e50=float(lat["EMA50"]); e200=float(lat["EMA200"])
    rsi_v=float(lat["RSI"]); vwap_v=float(lat["VWAP"]); st_val=float(lat["Supertrend"])
    st_dir=str(lat["SupertrendDir"])
    atr_v=float(lat["ATR"]) if not np.isnan(float(lat["ATR"])) else price*0.02
    macd_v=float(lat["MACD"]); macd_s=float(lat["MACD_Signal"])
    bb_u=float(lat["BB_Upper"]); bb_l=float(lat["BB_Lower"]); bb_m=float(lat["BB_Mid"])
    vol=float(lat["Volume"]); vol_avg=float(lat.get("Vol_EMA20",vol))
    bb_pos=((price-bb_l)/(bb_u-bb_l)*100) if (bb_u-bb_l)>0 else 50

    s1,s2,s3,r1,r2,r3 = get_key_levels(df)
    near_resistance = (r1-price)/price < 0.03

    fund_score, fund_reasons = score_fundamentals(info)
    tech_score, trend_bias, tech_vals, tech_reasons = score_technicals(df)
    risk_score, risk_reasons, vol_daily, max_dd = score_risk(df, atr_v, price)

    # Entry/exit
    if trend_bias=="Bullish":
        entry_low  = min(e20,e50)*0.999; entry_high = min(price*1.005, e20*1.01)
        stop_loss  = max(s1-atr_v*0.5, price-atr_v*2.0)
    else:
        entry_low  = price*0.99; entry_high = price*1.005
        stop_loss  = price - atr_v*1.5

    risk_per_sh = max(price-stop_loss, atr_v*0.5)
    target_1    = price + risk_per_sh*2.0
    target_2    = price + risk_per_sh*3.5
    rr_1        = (target_1-price)/risk_per_sh if risk_per_sh>0 else 0

    plan_score, plan_reasons = score_plan(trend_bias, sentiment_score, e20, e50)
    ee_score, ee_reasons     = score_entry_exit(trend_bias, rr_1, entry_high, stop_loss, target_1, target_2, price)
    mind_score, mind_reasons = score_mindset(vol_daily, rsi_v, near_resistance)

    scores = {"Fundamentals":fund_score,"Technicals":tech_score,
              "Risk":risk_score,"Plan":plan_score,"Entry/Exit":ee_score,"Mindset":mind_score}
    overall = round(sum(scores.values())/len(scores),2)

    all_reasons = {"Fundamentals":fund_reasons,"Technicals":tech_reasons,
                   "Risk":risk_reasons,"Plan":plan_reasons,
                   "Entry/Exit":ee_reasons,"Mindset":mind_reasons}

    return {
        "price":price,"rsi":rsi_v,"trend_bias":trend_bias,
        "atr":atr_v,"atr_pct":atr_v/price*100,"vol_daily":vol_daily,
        "support_watch":s1,"resistance_watch":r1,
        "entry_low":entry_low,"entry_high":entry_high,
        "stop_loss":stop_loss,"target_1":target_1,"target_2":target_2,"rr":rr_1,
        "scores":scores,"overall_score":overall,"all_reasons":all_reasons,
        "levels":(s1,s2,s3,r1,r2,r3),
        "market_cap":info.get("marketCap"),"sector":info.get("sector",""),
        "long_name":info.get("longName",ticker),
        "ema9":e9,"ema20":e20,"ema50":e50,"ema200":e200,
        "vwap":vwap_v,"bb_pos":bb_pos,"bb_u":bb_u,"bb_l":bb_l,"bb_m":bb_m,
        "st_dir":st_dir,"st_val":st_val,"max_dd":max_dd,
        "rev_growth":info.get("revenueGrowth",0) or 0,
        "eps_growth":info.get("earningsGrowth",0) or 0,
        "inst_pct":info.get("heldPercentInstitutions",0) or 0,
        "float_shares":info.get("floatShares",0) or 0,
        "vol_ratio":vol/vol_avg if vol_avg>0 else 1.0,
        "vol":vol,"vol_avg":vol_avg,
        "pe":info.get("trailingPE"),"fwd_pe":info.get("forwardPE"),
        "fund_score":fund_score,"tech_score":tech_score,"risk_score":risk_score,
        "near_resistance":near_resistance,
    }

# ══════════════════════════════════════════════════════════════
#  ADVANCED ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════
def compute_volume_intelligence(df, a):
    price    = a["price"]; vol = a["vol"]; vol_avg = a["vol_avg"]
    rvol     = vol/vol_avg if vol_avg>0 else 1.0
    spike    = rvol > 2.0
    # Accumulation vs distribution using price-close position in bar
    recent   = df.tail(5)
    close_pos= [(float(r["Close"])-float(r["Low"]))/(float(r["High"])-float(r["Low"])+0.0001) for _,r in recent.iterrows()]
    avg_pos  = sum(close_pos)/len(close_pos)
    accum    = avg_pos > 0.6  # closing near highs = accumulation
    distrib  = avg_pos < 0.4
    if rvol >= 2.0:
        participation = "High Participation"; participation_color = "green"
    elif rvol >= 0.8:
        participation = "Normal Participation"; participation_color = "gold"
    else:
        participation = "Weak Participation"; participation_color = "red"
    # Interpretation
    if rvol>=1.5 and a["trend_bias"]=="Bullish":
        note = f"High RVOL {rvol:.1f}x confirms bullish move — strong institutional participation."
    elif rvol>=1.5 and a["trend_bias"]=="Bearish":
        note = f"High RVOL {rvol:.1f}x on bearish day — distribution, selling pressure real."
    elif rvol<0.7:
        note = f"Low RVOL {rvol:.1f}x — weak conviction. Move may not sustain without volume follow-through."
    else:
        note = f"RVOL {rvol:.1f}x — normal participation. Watch for volume expansion to confirm next move."
    if accum: note += " Candles closing near highs = accumulation bias."
    if distrib: note += " Candles closing near lows = distribution bias."
    return {"vol":vol,"vol_avg":vol_avg,"rvol":rvol,"spike":spike,
            "participation":participation,"participation_color":participation_color,
            "accum_distrib":"Accumulation" if accum else ("Distribution" if distrib else "Neutral"),
            "note":note}

def compute_volatility_context(df, a):
    price   = a["price"]; atr_v = a["atr"]; atr_pct = a["atr_pct"]
    # Today's move vs ATR
    today_high = float(df["High"].iloc[-1]); today_low = float(df["Low"].iloc[-1])
    today_move = (today_high - today_low) / price * 100
    atr_used_pct = (today_move / atr_pct * 100) if atr_pct > 0 else 0
    # ATR compression: ATR vs 20-period average ATR
    atr_series = df["ATR"].dropna()
    atr_avg20  = float(atr_series.tail(20).mean()) if len(atr_series)>=20 else atr_v
    compression= atr_v < atr_avg20 * 0.75
    expansion  = atr_v > atr_avg20 * 1.25
    # Volatility state
    if atr_pct < 1.5:   vol_state = "Low"; vol_color = "green"
    elif atr_pct < 3.0: vol_state = "Medium"; vol_color = "gold"
    else:               vol_state = "High"; vol_color = "red"
    # Expected daily range
    exp_up   = price + atr_v
    exp_down = price - atr_v
    # Note
    if atr_used_pct > 85:
        note = f"Price has already moved {atr_used_pct:.0f}% of its daily ATR. Avoid chasing — limited range remaining today."
    elif compression:
        note = "ATR compression detected — volatility squeeze. Watch for potential expansion breakout."
    elif expansion:
        note = f"Volatility expanding. ATR {atr_pct:.1f}% above average — wider stops needed, reduce size."
    else:
        note = f"Normal volatility. ATR ${atr_v:.2f} ({atr_pct:.1f}%) is within historical range."
    return {"atr":atr_v,"atr_pct":atr_pct,"today_move_pct":today_move,
            "atr_used_pct":atr_used_pct,"vol_state":vol_state,"vol_color":vol_color,
            "exp_up":exp_up,"exp_down":exp_down,"compression":compression,"expansion":expansion,
            "note":note}

def compute_institutional_positioning(a, df):
    price   = a["price"]; vwap_v = a["vwap"]
    vwap_dev = (price-vwap_v)/vwap_v*100
    if vwap_dev > 3:
        vwap_bias = "Far Above VWAP"; vwap_color = "red"
        note = f"Price {vwap_dev:+.1f}% above VWAP — extended. Risk of mean reversion toward ${vwap_v:.2f}."
    elif vwap_dev > 0.5:
        vwap_bias = "Above VWAP"; vwap_color = "green"
        note = f"Price {vwap_dev:+.1f}% above VWAP (${vwap_v:.2f}) — intraday institutional bias bullish."
    elif vwap_dev > -0.5:
        vwap_bias = "At VWAP"; vwap_color = "gold"
        note = f"Price at VWAP — decision zone. Break above = bullish; rejection = bearish."
    elif vwap_dev > -3:
        vwap_bias = "Below VWAP"; vwap_color = "red"
        note = f"Price {vwap_dev:.1f}% below VWAP — weak intraday structure. Sellers in control."
    else:
        vwap_bias = "Far Below VWAP"; vwap_color = "red"
        note = f"Price {vwap_dev:.1f}% below VWAP (${vwap_v:.2f}) — deep discount zone. Potential snap-back risk."
    inst_pct = a.get("inst_pct",0)*100
    if inst_pct>70: inst_note = f"{inst_pct:.0f}% institutional — high smart money conviction, supply controlled."
    elif inst_pct>50: inst_note = f"{inst_pct:.0f}% institutional — moderate institutional interest."
    elif inst_pct>0: inst_note = f"{inst_pct:.0f}% institutional — lower conviction, more retail-driven."
    else: inst_note = "Institutional data unavailable."
    return {"vwap":vwap_v,"vwap_dev":vwap_dev,"vwap_bias":vwap_bias,
            "vwap_color":vwap_color,"inst_pct":inst_pct,"note":note,"inst_note":inst_note}

def compute_market_structure(df):
    close = df["Close"].values; high = df["High"].values; low = df["Low"].values
    n     = len(close)
    if n < 20: return {"structure":"Insufficient data","phase":"Unknown","note":"Need more bars."}
    # Find recent swing highs/lows (5-bar lookback)
    lb = 5
    swing_highs = [float(high[i]) for i in range(lb,n-lb) if high[i]==max(high[i-lb:i+lb+1])]
    swing_lows  = [float(low[i])  for i in range(lb,n-lb) if low[i] ==min(low[i-lb: i+lb+1])]
    # Check last 2 swings
    hh = len(swing_highs)>=2 and swing_highs[-1]>swing_highs[-2]
    lh = len(swing_highs)>=2 and swing_highs[-1]<swing_highs[-2]
    hl = len(swing_lows)>=2  and swing_lows[-1] >swing_lows[-2]
    ll = len(swing_lows)>=2  and swing_lows[-1] <swing_lows[-2]
    if hh and hl:   structure="Uptrend"; struct_color="green"
    elif lh and ll: structure="Downtrend"; struct_color="red"
    elif lh and hl: structure="Range / Squeeze"; struct_color="gold"
    else:           structure="Transitional"; struct_color="gold"
    # Phase detection
    price = float(close[-1])
    ema20 = float(df["EMA20"].iloc[-1]); ema50 = float(df["EMA50"].iloc[-1])
    recent_chg = (price - float(close[-10]))/float(close[-10])*100 if n>=10 else 0
    if abs(recent_chg)>8 and structure=="Uptrend": phase="Impulse Move"
    elif abs(recent_chg)<2 and structure=="Uptrend": phase="Consolidation"
    elif price<ema20 and structure=="Uptrend": phase="Pullback"
    elif structure=="Downtrend": phase="Downtrend / Distribution"
    else: phase="Consolidation"
    # BOS detection
    bos = ""
    if ll and len(swing_lows)>=2:
        prev_support = swing_lows[-2]
        if price < prev_support*0.99: bos="⚠️ Break of Structure (bearish BOS detected)"
    elif hh and len(swing_highs)>=2:
        prev_res = swing_highs[-2]
        if price > prev_res*1.01: bos="✅ Break of Structure (bullish BOS — potential breakout)"
    note_map = {
        "Uptrend": "Higher highs and higher lows confirm uptrend structure — trend following setups preferred.",
        "Downtrend": "Lower highs and lower lows = downtrend. Avoid longs; look for short entries on bounces.",
        "Range / Squeeze": "Price in range. Trade the extremes or wait for a breakout.",
        "Transitional": "Mixed structure — no clear HH/HL or LH/LL pattern. Stand aside until confirmation.",
    }
    return {"structure":structure,"struct_color":struct_color,"phase":phase,
            "hh":hh,"lh":lh,"hl":hl,"ll":ll,"bos":bos,
            "note":note_map.get(structure,"Structure unclear.")}

def compute_momentum_score(df, a, mtf):
    price = a["price"]; rsi_v = a["rsi"]
    macd_v = float(df["MACD"].iloc[-1]); macd_s = float(df["MACD_Signal"].iloc[-1])
    macd_h = float(df["MACD_Hist"].iloc[-1])
    macd_h_prev = float(df["MACD_Hist"].iloc[-2]) if len(df)>1 else 0
    e9  = float(df["EMA9"].iloc[-1]); e20 = float(df["EMA20"].iloc[-1])
    e9_prev  = float(df["EMA9"].iloc[-5])  if len(df)>5  else e9
    e20_prev = float(df["EMA20"].iloc[-5]) if len(df)>5  else e20
    # Slopes
    e9_slope  = (e9-e9_prev)/e9_prev*100 if e9_prev>0 else 0
    e20_slope = (e20-e20_prev)/e20_prev*100 if e20_prev>0 else 0
    # Price acceleration
    ret5  = (price - float(df["Close"].iloc[-5]))/float(df["Close"].iloc[-5])*100 if len(df)>5 else 0
    ret10 = (price - float(df["Close"].iloc[-10]))/float(df["Close"].iloc[-10])*100 if len(df)>10 else 0
    accel = ret5 - ret10/2  # Recent 5 days vs prior 5 days
    # Score
    pts = 0
    if rsi_v > 50: pts += 20
    if rsi_v > 60: pts += 10
    if macd_v > macd_s: pts += 20
    if macd_h > 0 and macd_h > macd_h_prev: pts += 15  # accelerating MACD
    if e9_slope > 0: pts += 10
    if e20_slope > 0: pts += 10
    if accel > 0: pts += 10
    if mtf["alignment_score"] >= 65: pts += 5
    pts = min(100, max(0, pts))
    if pts >= 70:   classification = "Strong"; mom_color = "green"
    elif pts >= 45: classification = "Moderate"; mom_color = "gold"
    else:           classification = "Weak"; mom_color = "red"
    if pts>=70: note="Momentum strong across multiple indicators. Trend likely to continue near-term."
    elif pts>=45: note="Moderate momentum. Setup constructive but lacks conviction. Watch for MACD cross confirmation."
    else: note="Weak momentum despite possibly bullish price structure. Do not fight the momentum gauge."
    return {"score":pts,"classification":classification,"mom_color":mom_color,
            "rsi":rsi_v,"macd_bull":macd_v>macd_s,"e9_slope":e9_slope,
            "e20_slope":e20_slope,"accel":accel,"note":note}

def compute_risk_profile(a):
    vol = a["vol_daily"]; atr_pct = a["atr_pct"]; max_dd = a["max_dd"]
    if vol<1.5 and atr_pct<2: risk_class="Low Risk"; risk_color="green"; behavior="Smooth trend — mechanical stops work well."
    elif vol<3 and atr_pct<4: risk_class="Medium Risk"; risk_color="gold"; behavior="Moderate choppiness — ATR-based stops recommended."
    else:                     risk_class="High Risk"; risk_color="red"; behavior="Aggressive mover — reduce size, use wider stops or options."
    return {"risk_class":risk_class,"risk_color":risk_color,"vol_daily":vol,
            "atr_pct":atr_pct,"max_dd":max_dd,"behavior":behavior}

def compute_setup_quality(a, vol_intel, mom, struct, catalyst, mtf, opt_data):
    pts = 0; breakdown = []
    # MTF alignment (25pts)
    align = mtf["alignment_score"]
    mtf_pts = int(align/100*25); pts += mtf_pts
    breakdown.append(f"MTF Alignment {align}/100 → +{mtf_pts}pts")
    # Volume (15pts)
    rvol = vol_intel["rvol"]
    vol_pts = min(15, int(rvol*7.5)) if rvol>=0.8 else 0; pts += vol_pts
    breakdown.append(f"Volume RVOL {rvol:.1f}x → +{vol_pts}pts")
    # R:R (15pts)
    rr = a["rr"]
    rr_pts = 15 if rr>=2.5 else (12 if rr>=2.0 else (8 if rr>=1.5 else 0)); pts += rr_pts
    breakdown.append(f"R:R {rr:.2f}x → +{rr_pts}pts")
    # Event risk (15pts)
    er = catalyst["event_risk"]
    er_pts = 15 if er=="Low" else (5 if er=="Moderate" else 0); pts += er_pts
    breakdown.append(f"Event Risk {er} → +{er_pts}pts")
    # Momentum (15pts)
    mom_pts = int(mom["score"]/100*15); pts += mom_pts
    breakdown.append(f"Momentum {mom['score']}/100 → +{mom_pts}pts")
    # Structure (10pts)
    st_pts = 10 if struct["structure"]=="Uptrend" else (5 if struct["structure"] in ["Range / Squeeze","Transitional"] else 0)
    pts += st_pts; breakdown.append(f"Structure {struct['structure']} → +{st_pts}pts")
    # Distance from support (5pts)
    s1 = a["levels"][0]; price = a["price"]
    near_supp = (price-s1)/price < 0.05
    sp_pts = 5 if near_supp else 0; pts += sp_pts
    if near_supp: breakdown.append(f"Near support ${s1:.2f} → +{sp_pts}pts")
    # Volatility suitability (pass/fail — deduct if very high)
    if a["atr_pct"]>5: pts = max(0, pts-10); breakdown.append(f"High ATR {a['atr_pct']:.1f}% → -10pts")
    pts = min(100, max(0, pts))
    if pts>=85:   grade="A+"; grade_color="#26a69a"
    elif pts>=75: grade="A";  grade_color="#26a69a"
    elif pts>=60: grade="B";  grade_color="#f5c842"
    elif pts>=45: grade="C";  grade_color="#f5c842"
    else:         grade="D";  grade_color="#ef5350"
    if pts>=75: note="Strong setup — multiple factors align. Act with defined risk."
    elif pts>=60: note="Good setup — most factors positive. Size appropriately."
    elif pts>=45: note="Average setup — some positive, some negative. Consider waiting."
    else: note="Weak setup — too many negatives. Better to stay out."
    return {"score":pts,"grade":grade,"grade_color":grade_color,"breakdown":breakdown,"note":note}

def classify_trade_setup(a, struct, vol_intel, mom):
    price=a["price"]; e20=a["ema20"]; e50=a["ema50"]
    s1,s2,s3,r1,r2,r3 = a["levels"]
    near_r = (r1-price)/price < 0.03
    near_s = (price-s1)/price < 0.03
    bb_pos = a["bb_pos"]
    trend  = a["trend_bias"]
    st_dir = a["st_dir"]
    rsi_v  = a["rsi"]
    hh = struct.get("hh",False); lh = struct.get("lh",False)
    hl = struct.get("hl",False); ll = struct.get("ll",False)
    if trend=="Bullish" and near_r and vol_intel["rvol"]>=1.3:
        setup="Breakout"; note=f"Price ${price:.2f} approaching resistance ${r1:.2f} with strong volume {vol_intel['rvol']:.1f}x. Watch for close above ${r1:.2f} on high volume to confirm."
    elif trend=="Bullish" and near_s and st_dir=="up" and rsi_v<65:
        setup="Pullback"; note=f"Price pulling back to support ${s1:.2f} in an uptrend. Ideal entry zone for swing buy with stop below ${s1*(0.98):.2f}."
    elif (ll or lh) and rsi_v<30:
        setup="Reversal"; note=f"Potential reversal — price in downtrend but RSI {rsi_v:.0f} oversold. Wait for ST to flip UP before entering."
    elif struct["structure"]=="Range / Squeeze":
        setup="Range Trade"; note=f"Price consolidating between ${s1:.2f} support and ${r1:.2f} resistance. Trade the range or wait for breakout."
    elif trend=="Bullish" and hh and hl and st_dir=="up" and mom["score"]>=50:
        setup="Trend Continuation"; note=f"Uptrend intact (HH+HL) with Supertrend UP and momentum {mom['score']}/100. Buy dips to EMA20 (${e20:.2f})."
    else:
        setup="No Valid Setup"; note="Conflicting signals across structure, momentum, and trend. Stand aside — wait for clearer setup."
    return {"setup":setup,"note":note}

def compute_expected_move(a, opt_data):
    price=a["price"]; atr_v=a["atr"]; atr_pct=a["atr_pct"]
    iv = opt_data.get("iv")
    # ATR-based
    exp_up_atr   = price + atr_v*1.5
    exp_down_atr = price - atr_v*1.5
    # IV-based (if available): 1-week move = IV * price * sqrt(5/252)
    if iv:
        iv_dec = iv/100
        weekly_move = iv_dec * price * (5/252)**0.5
        exp_up_iv   = price + weekly_move
        exp_down_iv = price - weekly_move
        note_iv = f"Options IV {iv:.1f}% implies ±${weekly_move:.2f} weekly move."
    else:
        exp_up_iv = exp_up_atr; exp_down_iv = exp_down_atr
        note_iv = "IV-based estimate unavailable — using ATR method only."
    # Check vs targets
    t1=a["target_1"]; t2=a["target_2"]
    t1_realistic = t1 <= exp_up_atr*1.1
    t2_realistic = t2 <= exp_up_atr*1.5
    t1_note = "Target 1 fits within normal expected range ✅" if t1_realistic else "Target 1 exceeds typical ATR move — may require strong momentum ⚠️"
    t2_note = "Target 2 is ambitious — plan to take partial profit at T1 ⚠️" if not t2_realistic else "Target 2 achievable in extended moves ✅"
    return {"exp_up_atr":exp_up_atr,"exp_down_atr":exp_down_atr,
            "exp_up_iv":exp_up_iv,"exp_down_iv":exp_down_iv,
            "atr_pct":atr_pct,"note_iv":note_iv,
            "t1_realistic":t1_realistic,"t2_realistic":t2_realistic,
            "t1_note":t1_note,"t2_note":t2_note}

def compute_failure_risks(a, vol_intel, mom, struct, catalyst, mtf, opt_data, meta):
    risks = []
    # Conflicting higher TF
    if mtf["alignment_score"] < 50:
        risks.append(("🔴 MTF Conflict", f"Alignment {mtf['alignment_score']}/100 — lower timeframes bullish but higher TFs bearish. High whipsaw risk."))
    # Volume
    if vol_intel["rvol"] < 0.7:
        risks.append(("🟡 Weak Volume", f"RVOL {vol_intel['rvol']:.1f}x — low participation. Move may fail without volume support."))
    # Momentum
    if mom["score"] < 40:
        risks.append(("🟡 Weak Momentum", f"Momentum score {mom['score']}/100 — underlying strength doesn't support current price action."))
    # Event risk
    if catalyst["event_risk"] == "High":
        risks.append(("🔴 Earnings Risk", f"Earnings ≤5 days — binary event. IV elevated. Direction unpredictable. Avoid new positions."))
    elif catalyst["event_risk"] == "Moderate":
        risks.append(("🟡 Upcoming Earnings", f"Earnings in 6–14 days — use defined risk structures. Don't hold large unhedged position through earnings."))
    # R:R
    if a["rr"] < 1.5:
        risks.append(("🔴 Poor R:R", f"R:R {a['rr']:.2f}x — below 1.5x minimum. Improve entry or skip this trade."))
    # Overextension
    bb_pos = a.get("bb_pos", 50)
    if bb_pos > 85:
        risks.append(("🟡 Overextended", f"Price at upper Bollinger Band ({bb_pos:.0f}th percentile). Mean reversion risk elevated."))
    # Near major resistance
    s1,s2,s3,r1,r2,r3 = a["levels"]; price=a["price"]
    if (r1-price)/price < 0.03:
        risks.append(("🟡 Near Resistance", f"Price within 3% of resistance ${r1:.2f}. Needs to break and hold above to continue."))
    # Stop width
    pct_stop = (price-a["stop_loss"])/price*100
    if pct_stop < 1.5:
        risks.append(("🟡 Stop Too Tight", f"Stop only {pct_stop:.1f}% below — likely to trigger on normal noise. ATR-based minimum: {a['atr_pct']:.1f}%."))
    elif pct_stop > 12:
        risks.append(("🟡 Stop Too Wide", f"Stop {pct_stop:.1f}% below — forces very small position size. Consider tighter structure."))
    # Stale data
    if meta:
        lc = meta.get("last_candle")
        if lc:
            try:
                lc2 = pd.Timestamp(lc)
                if lc2.tzinfo: lc2 = lc2.tz_localize(None)
                age_min = int((datetime.now()-lc2).total_seconds()/60)
                if age_min > 1440:
                    risks.append(("⚪ Stale Data", f"Last bar is {age_min//60}h old — ensure you have latest price before trading."))
            except Exception: pass
    # High IV
    iv = opt_data.get("iv")
    if iv and iv > 60:
        risks.append(("🟡 High IV", f"IV {iv:.1f}% elevated — options debit expensive. Use spreads or stick to stock."))
    if not risks:
        risks.append(("✅ No Major Risks", "Setup passes all failure checks. Execute with defined risk and honor your stop."))
    return risks

# ══════════════════════════════════════════════════════════════
#  AI COMMENTARY
# ══════════════════════════════════════════════════════════════
def generate_ai_commentary(payload, provider="none", model="openrouter/auto"):
    if provider=="none" or not st.session_state.get("ai_enabled", True):
        return _expert_python_summary(payload), None
    result, err = _call_openrouter(_build_ai_prompt(payload), model)
    if result: return result, None
    return _expert_python_summary(payload), f"⚠️ OpenRouter error: {err}. Python analysis shown."

def _build_ai_prompt(p):
    return f"""You are a senior portfolio manager. Analyze {p.get('ticker')} using ONLY the data below. Be specific and data-driven.

Price: ${p.get('price',0):.2f} | Trend: {p.get('trend_bias')} | RSI: {p.get('rsi',50):.1f}
Score: {p.get('overall_score',3)}/5 | MTF: {p.get('alignment_score',50)}/100 | Bias: {p.get('primary_bias')}
Entry: ${p.get('entry_low',0):.2f}–${p.get('entry_high',0):.2f} | Stop: ${p.get('stop_loss',0):.2f} | T1: ${p.get('target_1',0):.2f} | T2: ${p.get('target_2',0):.2f} | R:R: {p.get('rr',0):.2f}x
Vehicle: {p.get('options_vehicle')} | IV: {p.get('iv','N/A')} | Event Risk: {p.get('event_risk')}
Sentiment: {p.get('sentiment_label')} ({p.get('sentiment_score',50)}/100)
Fund Score: {p.get('fund_score',3)}/5 | Tech Score: {p.get('tech_score',3)}/5
Rev Growth: {(p.get('rev_growth',0) or 0)*100:.1f}% | EPS Growth: {(p.get('eps_growth',0) or 0)*100:.1f}%
ATR%: {p.get('atr_pct',2):.2f}% | Max DD: {p.get('max_dd',-20):.1f}%
Support: ${p.get('support',0):.2f} | Resistance: ${p.get('resistance',0):.2f}
Setup Quality: {p.get('setup_quality_score',50)}/100 grade {p.get('setup_grade','B')}
Momentum: {p.get('momentum_score',50)}/100 {p.get('momentum_class','')}
Market Structure: {p.get('structure','Unknown')} | Phase: {p.get('phase','Unknown')}
Setup Classification: {p.get('setup_type','Unknown')}

Respond with EXACTLY these 8 sections:

### 📊 Executive Summary
[2-3 sentences: overall assessment, conviction level, actionability]

### 🔬 Technical Analysis
[Price structure, EMA stack, MACD, RSI — use exact numbers above]

### 💼 Fundamental Assessment
[Revenue/EPS growth quality, margins, valuation context]

### 🌐 Sentiment & Catalyst
[News sentiment, event risk assessment, key upcoming catalyst]

### ⚡ Risk Assessment
[ATR risk, earnings proximity, max drawdown, position sizing guidance]

### 🎯 Specific Trade Recommendation
[Exact entry trigger, stop rationale, T1/T2 exits, time horizon]

### 🚨 Key Catalysts & Invalidation
[2-3 catalysts, exact invalidation price, what changes the view]

### 🧠 Trader Psychology Note
[Common trap for this setup, one discipline rule, contrarian consideration]"""

def _expert_python_summary(p):
    ticker=p.get("ticker","?"); trend=p.get("trend_bias","Neutral"); rsi=p.get("rsi",50)
    align=p.get("alignment_score",50); score=p.get("overall_score",3)
    s1=p.get("support",0); r1=p.get("resistance",0); stop=p.get("stop_loss",0)
    t1=p.get("target_1",0); t2=p.get("target_2",0); rr=p.get("rr",0)
    vehicle=p.get("options_vehicle","Wait"); event=p.get("event_risk","Low")
    sent_sc=p.get("sentiment_score",50); sent_lb=p.get("sentiment_label","Neutral")
    price=p.get("price",0); ema20=p.get("ema20",0); ema50=p.get("ema50",0); ema200=p.get("ema200",0)
    rev_g=(p.get("rev_growth",0) or 0)*100; eps_g=(p.get("eps_growth",0) or 0)*100
    fund_sc=p.get("fund_score",3); tech_sc=p.get("tech_score",3)
    momentum=p.get("momentum_score",50); structure=p.get("structure","Unknown")
    quality=p.get("setup_quality_score",50); grade=p.get("setup_grade","B")
    conviction="HIGH" if score>=4 else ("MODERATE" if score>=3 else "LOW")

    return f"""### 📊 Executive Summary
**{ticker}** presents a **{conviction} conviction {trend.lower()}** setup with overall score **{score}/5** and MTF alignment **{align}/100**.
Setup quality grade **{grade}** ({quality}/100). {'Ready to trade with defined risk.' if quality>=60 else 'Wait for better conditions.'}
Sentiment {sent_lb} at {sent_sc}/100 provides {'tailwind' if sent_sc>60 else 'headwind' if sent_sc<40 else 'neutral backdrop'}.

### 🔬 Technical Analysis
Price **${price:.2f}** is {'above' if price>ema20 else 'below'} EMA20 (${ema20:.2f}) and {'above' if price>ema50 else 'below'} EMA50 (${ema50:.2f}).
RSI **{rsi:.1f}** — {'healthy bull zone, room to run' if 45<rsi<65 else 'overbought, pullback risk' if rsi>70 else 'oversold, bounce zone' if rsi<30 else 'neutral'}.
Technical score {tech_sc}/5. Key support ${s1:.2f}, resistance ${r1:.2f}.

### 💼 Fundamental Assessment
Fundamental score **{fund_sc}/5**. Revenue growth {rev_g:.1f}% YoY, EPS trajectory {eps_g:.1f}%.
{'Strong fundamental backdrop supports the technical setup.' if fund_sc>=3.5 else 'Weak fundamentals — trade technically, not as a long-term hold.'}

### 🌐 Sentiment & Catalyst
Sentiment **{sent_lb}** ({sent_sc}/100). Event risk **{event}**.
{'Earnings approaching — use defined risk structures, avoid holding through binary event.' if event!='Low' else 'Clean catalyst window — no known binary events near-term.'}

### ⚡ Risk Assessment
Momentum score {momentum}/100. Structure: {structure}.
Stop at **${stop:.2f}** — {'honor this level absolutely, no averaging down' if True else ''}.
Max historical drawdown context — size positions for max 1-2% account loss if stop hit.

### 🎯 Specific Trade Recommendation
{'Long' if trend=='Bullish' else 'Short' if trend=='Bearish' else 'Wait'} | Vehicle: {vehicle}
Entry: ${p.get('entry_low',0):.2f}–${p.get('entry_high',0):.2f} | Stop: ${stop:.2f} | T1: ${t1:.2f} | T2: ${t2:.2f} | R:R: {rr:.2f}x

### 🚨 Key Catalysts & Invalidation
Trade invalidated below **${stop:.2f}** on closing basis. {'Earnings binary event is primary risk.' if event!='Low' else 'Watch for macro events and sector rotation.'}
{'Bull case: price reclaims ${:.2f} resistance on volume.'.format(r1)} Bear case: close below EMA50 (${ema50:.2f}) signals trend shift.

### 🧠 Trader Psychology Note
{'Chasing risk — RSI extended, price near resistance. Wait for pullback to EMA20.' if rsi>68 else 'Knife-catching risk — do not average down. Let price come to your entry zone.' if trend=='Bearish' else 'Patience is the edge. Define entry, stop, target before touching the order button.'}
Write your trade plan BEFORE entering. If the setup changes, exit — do not hope."""

def generate_advanced_ai(adv_payload, provider="none", model="openrouter/auto"):
    if provider=="none" or not st.session_state.get("ai_enabled", True):
        return _advanced_python_summary(adv_payload), None
    prompt = f"""You are a professional trading analyst. Using ONLY the computed data below, provide Advanced Desk Notes.

TICKER: {adv_payload.get('ticker')}
Setup Quality: {adv_payload.get('setup_quality_score')}/100 Grade {adv_payload.get('setup_grade')}
Momentum: {adv_payload.get('momentum_score')}/100 {adv_payload.get('momentum_class')}
Market Structure: {adv_payload.get('structure')} | Phase: {adv_payload.get('phase')}
Volume: RVOL {adv_payload.get('rvol',1):.2f}x | {adv_payload.get('vol_participation')}
Volatility: ATR {adv_payload.get('atr_pct',2):.2f}% | State: {adv_payload.get('vol_state')} | Used: {adv_payload.get('atr_used_pct',0):.0f}% of daily range
VWAP: {adv_payload.get('vwap_bias')} | Deviation: {adv_payload.get('vwap_dev',0):+.2f}%
Setup Type: {adv_payload.get('setup_type')}
Failure Risks: {adv_payload.get('top_risks')}
Expected Move: Up ${adv_payload.get('exp_up',0):.2f} / Down ${adv_payload.get('exp_down',0):.2f}
Trend: {adv_payload.get('trend_bias')} | R:R: {adv_payload.get('rr',0):.2f}x | MTF: {adv_payload.get('mtf_align')}/100

Respond with EXACTLY these 7 sections (2-3 sentences each):

### 🏆 Overall Setup Quality
[Grade explanation — what earned or reduced the score]

### 📈 Strongest Bullish Factors
[Top 2-3 bullish factors from the data above]

### 📉 Biggest Risk Factors
[Top 2-3 risks — be specific with numbers]

### ⏰ Act Now or Wait?
[Clear recommendation with specific trigger]

### 🎯 Entry Trigger
[Exact price or condition that should trigger entry]

### 🛠️ Best Trade Style
[Breakout / Pullback / Wait / Avoid — and why]

### 🧘 Discipline Reminder
[One specific discipline rule for this exact setup]"""

    result, err = _call_openrouter(prompt, model, max_tokens=900)
    if result: return result, None
    return _advanced_python_summary(adv_payload), f"⚠️ OpenRouter: {err}"

def _advanced_python_summary(p):
    grade=p.get("setup_grade","B"); qs=p.get("setup_quality_score",50)
    mom=p.get("momentum_score",50); mom_class=p.get("momentum_class","Moderate")
    struct=p.get("structure","Unknown"); phase=p.get("phase","Unknown")
    rvol=p.get("rvol",1); vwap_bias=p.get("vwap_bias","At VWAP")
    setup_type=p.get("setup_type","No Valid Setup"); trend=p.get("trend_bias","Neutral")
    rr=p.get("rr",0); risks=p.get("top_risks","None identified")
    return f"""### 🏆 Overall Setup Quality
Setup grades **{grade}** ({qs}/100). {'Multiple factors aligned — this meets the threshold for a defined-risk trade.' if qs>=65 else 'Setup has notable weaknesses — either wait or reduce size significantly.'}

### 📈 Strongest Bullish Factors
Market structure shows **{struct}** with phase **{phase}**. RVOL at {rvol:.1f}x indicates {'strong institutional participation.' if rvol>=1.5 else 'normal participation — watch for expansion.'} VWAP positioning: {vwap_bias}.

### 📉 Biggest Risk Factors
{risks if risks else 'Primary risks: event risk, momentum divergence, and stop placement.'} Momentum at {mom}/100 ({mom_class}).

### ⏰ Act Now or Wait?
{'Setup is actionable — execute at entry zone with full risk defined before order placement.' if qs>=65 else 'Wait for higher quality setup. Patience is a position.'}

### 🎯 Entry Trigger
{f'Pullback to EMA20 on declining volume, confirmed by a green candle close above EMA9.' if trend=='Bullish' else f'Bounce to EMA20 rejected by a red candle close.' if trend=='Bearish' else 'Wait for clear trend direction before defining entry trigger.'}

### 🛠️ Best Trade Style
**{setup_type}**. {'R:R {:.2f}x makes this worthwhile with proper sizing.'.format(rr) if rr>=1.5 else 'R:R {:.2f}x too low — improve entry before committing.'.format(rr)}

### 🧘 Discipline Reminder
Write ENTRY, STOP, and TARGET before touching the order button. If price does not reach your zone, you do not trade — that IS a trade decision, and a good one."""

# ══════════════════════════════════════════════════════════════
#  SIGNAL CHIPS
# ══════════════════════════════════════════════════════════════
def generate_signals(df, a, catalyst, mtf_align, info=None):
    chips = []; price=a["price"]; rsi_v=a["rsi"]
    e20=a["ema20"]; e50=a["ema50"]; e200=a["ema200"]
    st_dir=str(df.iloc[-1]["SupertrendDir"]); vol=a["vol"]; vol_avg=a["vol_avg"]
    vr=vol/vol_avg if vol_avg>0 else 1.0; bb_pos=a["bb_pos"]
    hi52=float(df["High"].tail(252).max()) if len(df)>=252 else float(df["High"].max())
    lo52=float(df["Low"].tail(252).min())  if len(df)>=252 else float(df["Low"].min())
    pct_hi=(price-hi52)/hi52*100; pct_lo=(price-lo52)/lo52*100
    if price>e20>e50>e200 and st_dir=="up": chips.append(("✅ Perfect Bull Stack","green"))
    elif price>e50 and price>e200 and st_dir=="up": chips.append(("✅ Above All MAs","green"))
    elif price<e50 and price<e200: chips.append(("⚠️ Below Key MAs","red"))
    if pct_hi>-5: chips.append(("📈 Near 52W High","green"))
    elif pct_lo<15: chips.append(("📉 Near 52W Low","red"))
    if rsi_v>70:   chips.append(("⚠️ RSI Overbought","gold"))
    elif rsi_v<30: chips.append(("⚠️ RSI Oversold","red"))
    elif 45<rsi_v<65: chips.append(("✅ RSI Healthy","green"))
    if vr>1.5: chips.append((f"🔊 Volume {vr:.1f}x Avg","blue"))
    elif vr<0.5: chips.append(("🔇 Low Volume","gold"))
    if bb_pos>85: chips.append(("⚡ Upper BB Extended","gold"))
    elif bb_pos<15: chips.append(("🎯 Lower BB Oversold Zone","green"))
    if mtf_align>=70: chips.append((f"🎯 MTF Aligned {mtf_align}/100","green"))
    elif mtf_align<=35: chips.append((f"⚔️ MTF Conflict {mtf_align}/100","red"))
    dte=catalyst.get("days_to_earnings")
    if dte is not None and dte<=5: chips.append((f"🔴 Earnings in {dte}d","red"))
    elif dte is not None and dte<=14: chips.append((f"🟡 Earnings in {dte}d","gold"))
    chips.append(("🟢 Supertrend UP","green") if st_dir=="up" else ("🔴 Supertrend DOWN","red"))
    macd_v=float(df.iloc[-1]["MACD"]); macd_s=float(df.iloc[-1]["MACD_Signal"])
    macd_h=float(df.iloc[-1]["MACD_Hist"]); macd_hp=float(df.iloc[-2]["MACD_Hist"]) if len(df)>1 else 0
    if macd_v>macd_s and macd_h>0 and macd_h>macd_hp: chips.append(("📈 MACD Bullish Momentum","green"))
    elif macd_v<macd_s and macd_h<0 and macd_h<macd_hp: chips.append(("📉 MACD Bearish Momentum","red"))
    elif macd_v>macd_s: chips.append(("📊 MACD Above Signal","green"))
    else: chips.append(("📊 MACD Below Signal","red"))
    if info:
        inst=(info.get("heldPercentInstitutions") or 0)
        if inst>0.7: chips.append((f"🏛️ {inst*100:.0f}% Institutional","blue"))
    return chips

def render_chips(chips):
    html=""
    for label,color in chips: html+=f'<span class="chip chip-{color}">{label}</span>'
    return html

# ══════════════════════════════════════════════════════════════
#  MTF ENGINE
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def get_mtf_data(ticker):
    tfs=["5m","15m","1h","4h","1d","1wk"]; rows=[]; bull=0.0; cnt=0
    for tf in tfs:
        df_tf,_,err,_ = get_stock_data(ticker,tf)
        if err or df_tf is None or len(df_tf)<20:
            rows.append({"TF":tf,"EMA9":"—","EMA20":"—","RSI":"—","MACD":"—","ST":"—","Verdict":"⬜ No Data"})
            continue
        lat=df_tf.iloc[-1]; p=float(lat["Close"])
        e9=float(lat["EMA9"]); e20=float(lat["EMA20"])
        r=float(lat["RSI"]); mc=float(lat["MACD"]); ms=float(lat["MACD_Signal"])
        sd=str(lat["SupertrendDir"]); vw=float(lat["VWAP"]) if "VWAP" in df_tf.columns else None
        pts=0
        if p>e9: pts+=1
        if p>e20: pts+=1
        if mc>ms: pts+=1
        if sd=="up": pts+=1
        if r>50: pts+=1
        if vw and p>vw: pts+=1
        pct=pts/6
        if pct>=0.75:   v="🟢 Bullish"; bull+=1.0
        elif pct>=0.5:  v="🟡 Mild Bull"; bull+=0.5
        elif pct>=0.25: v="🟠 Mild Bear"
        else:           v="🔴 Bearish"
        cnt+=1
        rows.append({"TF":tf,"EMA9":"▲" if p>e9 else "▼","EMA20":"▲" if p>e20 else "▼",
                     "RSI":f"{r:.0f}","MACD":"▲" if mc>ms else "▼",
                     "ST":"▲" if sd=="up" else "▼","Verdict":v})
    align=int((bull/max(cnt,1))*100)
    if align>=75:   pb="Strong Bullish"; trig="Buy pullbacks to EMA9/20"; cf="Watch overbought RSI on 1d"
    elif align>=55: pb="Mild Bullish";   trig="Wait for 1h close above EMA20"; cf="Size down — some TF conflict"
    elif align>=45: pb="Neutral/Choppy"; trig="Avoid — wait for resolution"; cf="High whipsaw risk"
    elif align>=25: pb="Mild Bearish";   trig="Avoid longs; watch bounces"; cf="1d may still transition"
    else:           pb="Strong Bearish"; trig="Short on failed bounces to EMA20"; cf="Confirm with volume"
    return {"rows":rows,"alignment_score":align,"primary_bias":pb,"trigger":trig,"conflict_note":cf}

# ══════════════════════════════════════════════════════════════
#  CHART
# ══════════════════════════════════════════════════════════════
def build_chart(df, ticker, interval, show_bb=True, show_vwap=True, show_st=True,
                show_ema=True, show_macd=False):
    vc=np.where(df["Close"]>=df["Open"],"#26a69a","#ef5350")
    rows=3 if show_macd else 2; hs=[0.60,0.15,0.25] if show_macd else [0.74,0.26]
    fig=make_subplots(rows=rows,cols=1,shared_xaxes=True,vertical_spacing=0.01,row_heights=hs)
    fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],low=df["Low"],close=df["Close"],
        name="Price",increasing=dict(line=dict(color="#26a69a",width=1),fillcolor="#26a69a"),
        decreasing=dict(line=dict(color="#ef5350",width=1),fillcolor="#ef5350"),whiskerwidth=0),row=1,col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index,y=df["BB_Upper"],name="BB U",
            line=dict(color="#5c6bc0",width=0.8,dash="dot"),showlegend=False),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["BB_Lower"],name="BB L",
            line=dict(color="#5c6bc0",width=0.8,dash="dot"),fill="tonexty",
            fillcolor="rgba(92,107,192,0.05)",showlegend=False),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["BB_Mid"],name="BB Mid",
            line=dict(color="#5c6bc0",width=0.7,dash="dash"),showlegend=False),row=1,col=1)
    if show_ema:
        for cn,clr,lw,lbl in [("EMA9","#ffffff",1.0,"EMA 9"),("EMA20","#f5c542",1.0,"EMA 20"),
                                ("EMA50","#4da3ff",1.2,"EMA 50"),("EMA200","#b388ff",1.4,"EMA 200")]:
            fig.add_trace(go.Scatter(x=df.index,y=df[cn],mode="lines",
                name=lbl,line=dict(color=clr,width=lw)),row=1,col=1)
    if show_vwap:
        fig.add_trace(go.Scatter(x=df.index,y=df["VWAP"],mode="lines",name="VWAP",
            line=dict(color="#ff9800",width=1.5,dash="dash")),row=1,col=1)
    if show_st:
        fig.add_trace(go.Scatter(x=df.index,y=df["Supertrend"].where(df["SupertrendDir"]=="up"),
            mode="lines",name="ST Bull",line=dict(color="#26a69a",width=2)),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["Supertrend"].where(df["SupertrendDir"]=="down"),
            mode="lines",name="ST Bear",line=dict(color="#ef5350",width=2)),row=1,col=1)
    s1,s2,s3,r1,r2,r3=get_key_levels(df)
    for lvl,clr,lbl in [(s1,"#26a69a","S1"),(r1,"#ef5350","R1"),(s2,"#1a7a72","S2"),(r2,"#a33534","R2")]:
        fig.add_hline(y=lvl,line_dash="dot",line_color=clr,line_width=0.9,
                      annotation_text=f" {lbl}: ${lvl:.2f}",
                      annotation_font_color=clr,annotation_font_size=10,row=1,col=1)
    fig.add_trace(go.Bar(x=df.index,y=df["Volume"],name="Volume",marker_color=vc,showlegend=False),row=2,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["Vol_EMA20"],mode="lines",name="Vol EMA20",
        line=dict(color="#f5c542",width=1.0,dash="dot"),showlegend=False),row=2,col=1)
    if show_macd:
        mc_c=np.where(df["MACD_Hist"]>=0,"#26a69a","#ef5350")
        fig.add_trace(go.Bar(x=df.index,y=df["MACD_Hist"],name="MACD Hist",marker_color=mc_c,showlegend=False),row=3,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["MACD"],mode="lines",name="MACD",line=dict(color="#2962ff",width=1.2)),row=3,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["MACD_Signal"],mode="lines",name="Signal",line=dict(color="#ff6b35",width=1.0)),row=3,col=1)
        fig.add_hline(y=0,line_color="#363a45",line_width=0.5,row=3,col=1)
    fig.update_layout(
        title=dict(text=f"  {ticker}",font=dict(size=16,color="#d1d4dc"),x=0),
        template="plotly_dark",height=700 if show_macd else 580,
        dragmode="pan",hovermode="x unified",xaxis_rangeslider_visible=False,
        margin=dict(l=50,r=10,t=36,b=10),
        legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="left",x=0,font=dict(size=11)),
        plot_bgcolor="#131722",paper_bgcolor="#131722",font=dict(color="#d1d4dc",size=11))
    rb=[dict(bounds=["sat","mon"])]
    if interval in ("1m","5m","15m","1h"): rb.append(dict(bounds=[16,9.5],pattern="hour"))
    elif interval=="4h": rb.append(dict(bounds=[20,4],pattern="hour"))
    fig.update_xaxes(showgrid=True,gridcolor="#1f2937",gridwidth=0.5,zeroline=False,
                     showspikes=True,spikemode="across",spikecolor="#787b86",spikethickness=1,
                     showline=True,linecolor="#363a45",rangebreaks=rb,
                     rangeselector=dict(
                         buttons=[dict(count=5,label="5D",step="day",stepmode="backward"),
                                  dict(count=1,label="1M",step="month",stepmode="backward"),
                                  dict(count=3,label="3M",step="month",stepmode="backward"),
                                  dict(count=6,label="6M",step="month",stepmode="backward"),
                                  dict(count=1,label="1Y",step="year",stepmode="backward"),
                                  dict(step="all",label="All")],
                         bgcolor="#1e222d",activecolor="#2962ff",
                         font=dict(color="#d1d4dc",size=11),bordercolor="#363a45",borderwidth=1,
                     ) if len(df)>20 else None)
    fig.update_yaxes(showgrid=True,gridcolor="#1f2937",gridwidth=0.5,zeroline=False,
                     showline=True,linecolor="#363a45",tickformat=".2f",side="right")
    fig.update_yaxes(title_text="",row=2,col=1,tickformat=".2s")
    if show_macd: fig.update_yaxes(title_text="MACD",row=3,col=1)
    return fig

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
    if s>=4.5: return "🟢 Excellent"
    if s>=3.5: return "🟢 Strong"
    if s>=2.5: return "🟡 Average"
    return "🔴 Weak"

def score_color(s):
    if s>=3.5: return "#26a69a"
    if s>=2.5: return "#f5c842"
    return "#ef5350"

def render_data_integrity(meta, interval):
    if meta is None: return
    now=datetime.now(); ft=meta.get("fetch_time",now); lc=meta.get("last_candle")
    bar_age=None
    if lc:
        try:
            lc2=pd.Timestamp(lc)
            if lc2.tzinfo: lc2=lc2.tz_localize(None)
            bar_age=int((now-lc2).total_seconds()/60)
        except Exception: pass
    feed="🔄 Near-Realtime" if bar_age and bar_age<20 else ("⏱ Delayed" if bar_age and bar_age<60 else "📦 Historical")
    mkt="🟢 Open" if (now.weekday()<5 and 9<=now.hour<16) else "🔴 Closed"
    d1,d2,d3,d4,d5,d6,d7=st.columns(7)
    d1.metric("Source","Yahoo Finance"); d2.metric("Interval",interval)
    d3.metric("Fetched",ft.strftime("%H:%M:%S"))
    d4.metric("Last Bar",str(lc)[:16] if lc else "N/A")
    d5.metric("Bar Age",f"{bar_age}m" if bar_age else "N/A")
    d6.metric("Market",mkt); d7.metric("Feed",feed)

def load_journal():
    if not os.path.exists(JOURNAL_FILE): return pd.DataFrame(columns=JOURNAL_COLS)
    try:
        df=pd.read_csv(JOURNAL_FILE)
        for c in JOURNAL_COLS:
            if c not in df.columns: df[c]=""
        return df
    except Exception: return pd.DataFrame(columns=JOURNAL_COLS)

def save_trade(d):
    exists=os.path.exists(JOURNAL_FILE)
    with open(JOURNAL_FILE,"a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=JOURNAL_COLS)
        if not exists: w.writeheader()
        w.writerow({c:d.get(c,"") for c in JOURNAL_COLS})

def journal_analytics(df):
    if df.empty or "result" not in df.columns: return {}
    df=df.copy()
    df["pnl"]=pd.to_numeric(df["pnl"],errors="coerce").fillna(0)
    df["alignment_score"]=pd.to_numeric(df["alignment_score"],errors="coerce").fillna(0)
    total=len(df); wins=df[df["result"].str.lower()=="win"]; losses=df[df["result"].str.lower()=="loss"]
    wr=len(wins)/total*100 if total>0 else 0
    aw=wins["pnl"].mean() if not wins.empty else 0
    al=losses["pnl"].mean() if not losses.empty else 0
    gp=wins["pnl"].sum(); gl=abs(losses["pnl"].sum())
    pf=gp/gl if gl>0 else float("inf")
    exp=(wr/100*aw)+((1-wr/100)*al)
    insights=[]
    ha=df[df["alignment_score"]>=70]; la=df[df["alignment_score"]<50]
    if len(ha)>=2 and len(la)>=2:
        wrh=len(ha[ha["result"].str.lower()=="win"])/len(ha)*100
        wrl=len(la[la["result"].str.lower()=="win"])/len(la)*100
        if wrh>wrl+10:
            insights.append(f"✅ High-alignment trades win {wrh:.0f}% vs {wrl:.0f}% for low-alignment.")
    return {"total":total,"win_rate":wr,"avg_win":aw,"avg_loss":al,
            "total_pnl":df["pnl"].sum(),"profit_factor":pf,"expectancy":exp,"insights":insights}

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📊 Trading Desk")
    page=st.radio("nav",[
        "📈 Chart & Analysis","🔬 Advanced Analysis",
        "💼 Portfolio","👁️ Watchlist","📓 Journal",
    ],label_visibility="collapsed")
    st.divider()
    st.markdown("**🤖 AI Settings**")
    ai_provider=st.selectbox("AI Provider",["none","openrouter"],key="ai_provider")
    ai_model   =st.text_input("Model",value="openrouter/auto",key="ai_model")
    ai_enabled =st.toggle("Enable AI Comments",value=True,key="ai_enabled")
    key_ok=_get_or_key() is not None
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
    page=st.session_state.pop("page_override")

# ══════════════════════════════════════════════════════════════
#  PAGE 1: CHART & ANALYSIS
# ══════════════════════════════════════════════════════════════
if "📈" in page:
    st.markdown("## 📈 Chart & Analysis")
    c1,c2,c3=st.columns([3,1,1])
    with c1:
        default_tick=st.session_state.get("ticker_input","AVGO")
        ticker=st.text_input("Ticker",value=default_tick,placeholder="e.g. AVGO, META, NVDA").strip().upper()
    with c2:
        interval=st.selectbox("Interval",["1m","5m","15m","1h","4h","1d","1wk"],index=5)
    with c3:
        st.markdown("<br>",unsafe_allow_html=True)
        st.button("▶ Load",use_container_width=True)

    if not ticker: st.info("Enter a ticker symbol above."); st.stop()

    df,info,error,meta=get_stock_data(ticker,interval)
    if error or df is None: st.error(f"Could not load **{ticker}**: {error}"); st.stop()

    news=get_news(ticker); sent_score,sent_label=calculate_sentiment(news)
    a=analyze(df,info or {},ticker,sent_score)
    catalyst=get_catalyst_data(ticker); opt_data=get_options_data(ticker)
    mtf=get_mtf_data(ticker)
    signals=generate_signals(df,a,catalyst,mtf["alignment_score"],info or {})
    regime=get_market_regime()
    opt_vehicle,opt_reason=options_recommendation(a["trend_bias"],catalyst["event_risk"],
                                                   opt_data.get("iv"),a["rsi"],a["atr_pct"])
    cur=float(df["Close"].iloc[-1]); prev=float(df["Close"].iloc[-2]) if len(df)>1 else cur
    chg=cur-prev; chg_pct=chg/prev*100 if prev else 0

    # Data Integrity
    render_data_integrity(meta,interval)

    # Header — Ticker name + metrics
    company_name = a.get("long_name", ticker)
    if company_name and company_name != ticker:
        st.markdown(f"<span style='color:#787b86;font-size:13px'>{company_name}</span>", unsafe_allow_html=True)

    m1,m2,m3,m4,m5,m6=st.columns(6)
    m1.metric("Ticker",ticker)
    m2.metric("Price",f"${cur:.2f}",f"{chg:+.2f} ({chg_pct:+.2f}%)")
    m3.metric("RSI",f"{a['rsi']:.1f}","OB" if a['rsi']>70 else ("OS" if a['rsi']<30 else "✓"))
    m4.metric("Trend",a["trend_bias"])
    m5.metric("Score",f"{a['overall_score']:.1f}/5")
    m6.metric("Sentiment",f"{sent_score}/100 {sent_label}")
    st.divider()

    # Market Regime — with live SPY/QQQ prices
    st.markdown("### 🌐 Market Regime")
    r1c,r2c,r3c,r4c,r5c,r6c=st.columns(6)
    spy_p=regime.get("SPY_price"); spy_c=regime.get("SPY_chg",0)
    qqq_p=regime.get("QQQ_price"); qqq_c=regime.get("QQQ_chg",0)
    r1c.metric("SPY Trend",regime["SPY"])
    r2c.metric("SPY Price",f"${spy_p:.2f}" if spy_p else "N/A",
               f"{spy_c:+.2f}%" if spy_c else "")
    r3c.metric("QQQ Trend",regime["QQQ"])
    r4c.metric("QQQ Price",f"${qqq_p:.2f}" if qqq_p else "N/A",
               f"{qqq_c:+.2f}%" if qqq_c else "")
    r5c.metric("VIX",f"{regime['VIX']:.1f}" if regime["VIX"] else "N/A",
               "⚠️ High" if regime["VIX"] and regime["VIX"]>25 else ("Normal" if regime["VIX"] else ""))
    reg_color={"Risk-on":"#26a69a","Cautious Bull":"#f5c842","Risk-off":"#ef5350",
               "Extreme Fear":"#ef5350","Choppy":"#f5c842"}.get(regime["regime"],"#787b86")
    r6c.markdown(f"**Regime:**<br><span style='color:{reg_color};font-size:18px;font-weight:700'>{regime['regime']}</span>",unsafe_allow_html=True)
    st.divider()

    # Signal Chips
    st.markdown("### ⚡ Signals & Warnings")
    st.markdown(render_chips(signals),unsafe_allow_html=True)
    st.divider()

    # Chart Controls
    st.markdown("**Chart Overlays**")
    oc1,oc2,oc3,oc4,oc5=st.columns(5)
    with oc1: show_ema =st.checkbox("EMAs",value=True)
    with oc2: show_bb  =st.checkbox("Bollinger",value=False)
    with oc3: show_vwap=st.checkbox("VWAP",value=True)
    with oc4: show_st  =st.checkbox("Supertrend",value=True)
    with oc5: show_macd=st.checkbox("MACD Panel",value=False)
    st.caption("💡 Scroll=zoom · Drag=pan · Weekend gaps removed · Range buttons above chart")
    fig=build_chart(df,ticker,interval,show_bb,show_vwap,show_st,show_ema,show_macd)
    st.plotly_chart(fig,use_container_width=True,config={"scrollZoom":True,"displayModeBar":True,
        "modeBarButtonsToAdd":["drawline","drawopenpath","drawrect","eraseshape"],
        "toImageButtonOptions":{"format":"png","filename":f"{ticker}_chart"}})
    st.divider()

    # Catalyst
    st.markdown("### 🗓️ Catalyst Panel")
    st.markdown(f'<div class="{catalyst["risk_css"]}"><strong>{catalyst["risk_label"]}</strong></div>',unsafe_allow_html=True)
    st.markdown("")
    ca1,ca2,ca3,ca4=st.columns(4)
    ed=catalyst["earnings_date"]
    ca1.metric("Next Earnings",str(ed) if ed else "N/A",
               f"{catalyst['days_to_earnings']}d away" if catalyst["days_to_earnings"] is not None else "")
    ca2.metric("Event Risk",catalyst["event_risk"])
    ca3.metric("Ex-Div",str(catalyst["ex_div_date"]) if catalyst["ex_div_date"] else "N/A")
    at=catalyst["analyst_target"]
    if at: ca4.metric("Analyst Target",f"${at:.2f}",f"{(at-cur)/cur*100:+.1f}%")
    else:  ca4.metric("Analyst Target","N/A")
    st.divider()

    # Trade Plan + Key Metrics
    st.markdown("### 📌 Trade Plan & Key Metrics")
    tp_col,km_col=st.columns(2)
    s1,s2,s3,r1,r2,r3=a["levels"]
    pct_stop=(cur-a["stop_loss"])/cur*100 if cur>0 else 0
    with tp_col:
        st.markdown('<div class="card card-accent-blue">',unsafe_allow_html=True)
        st.markdown("**Trade Plan**")
        st.markdown(f"""<table class="trade-table">
<tr><td>Entry Zone</td><td style="color:#d1d4dc">${a['entry_low']:.2f} – ${a['entry_high']:.2f}</td></tr>
<tr><td>Stop Loss ({pct_stop:.1f}%)</td><td style="color:#ef5350">${a['stop_loss']:.2f}</td></tr>
<tr><td>Target 1 (+{((a['target_1']-cur)/cur*100):.1f}%)</td><td style="color:#26a69a">${a['target_1']:.2f}</td></tr>
<tr><td>Target 2 (+{((a['target_2']-cur)/cur*100):.1f}%)</td><td style="color:#26a69a">${a['target_2']:.2f}</td></tr>
<tr><td>Risk/Reward</td><td style="color:{'#26a69a' if a['rr']>=2 else '#f5c842'}">{a['rr']:.2f}:1</td></tr>
<tr><td>Support S1</td><td>${s1:.2f}</td></tr>
<tr><td>Support S2</td><td>${s2:.2f}</td></tr>
<tr><td>Resistance R1</td><td>${r1:.2f}</td></tr>
<tr><td>Resistance R2</td><td>${r2:.2f}</td></tr>
</table>""",unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
        # Plan commentary
        trend=a["trend_bias"]; rr=a["rr"]
        if trend=="Bullish" and rr>=2:
            st.markdown('<div class="signal-good">✅ Bullish trend with R:R ≥2x — constructive setup. Buy pullbacks to EMA20, not breakouts.</div>',unsafe_allow_html=True)
        elif trend=="Bullish" and rr<1.5:
            st.markdown('<div class="signal-alert">⚠️ Bullish trend but R:R {:.2f}x is too low. Improve entry or wait for a better pullback level.</div>'.format(rr),unsafe_allow_html=True)
        elif trend=="Bearish":
            st.markdown('<div class="signal-alert">🔴 Bearish trend — avoid new longs. Wait for price to reclaim EMA50 before considering any long setups.</div>',unsafe_allow_html=True)
        else:
            st.markdown('<div class="card" style="color:#787b86;font-size:12px">⚪ Neutral — No strong directional plan. Wait for confirmation from EMA stack and Supertrend alignment.</div>',unsafe_allow_html=True)

    with km_col:
        info_d=info or {}
        eps_g=(info_d.get("earningsGrowth") or 0)*100; rev_g=(info_d.get("revenueGrowth") or 0)*100
        inst=(info_d.get("heldPercentInstitutions") or 0)*100
        float_s=info_d.get("floatShares",0) or 0; vol_cur=float(df["Volume"].iloc[-1])
        vol_avg2=float(df["Vol_EMA20"].iloc[-1]) if "Vol_EMA20" in df.columns else vol_cur
        vr=vol_cur/vol_avg2 if vol_avg2>0 else 1.0
        def km_color(val,good,bad):
            return "#26a69a" if val>=good else ("#ef5350" if val<=bad else "#f5c842")
        st.markdown('<div class="card card-accent-gold">',unsafe_allow_html=True)
        st.markdown("**Key Metrics**")
        st.markdown(f"""<table class="trade-table">
<tr><td>EPS Qtr Growth</td><td style="color:{km_color(eps_g,15,-5)}">{eps_g:.1f}%</td></tr>
<tr><td>Revenue Growth</td><td style="color:{km_color(rev_g,15,0)}">{rev_g:.1f}%</td></tr>
<tr><td>Gross Margin</td><td style="color:{km_color((info_d.get('grossMargins') or 0)*100,50,25)}">{(info_d.get('grossMargins') or 0)*100:.1f}%</td></tr>
<tr><td>Net Margin</td><td style="color:{km_color((info_d.get('profitMargins') or 0)*100,15,0)}">{(info_d.get('profitMargins') or 0)*100:.1f}%</td></tr>
<tr><td>Volume vs Avg</td><td style="color:{km_color(vr,1.3,0.7)}">{vr:.2f}x</td></tr>
<tr><td>Institutional %</td><td style="color:{km_color(inst,65,30)}">{inst:.1f}%</td></tr>
<tr><td>Float Shares</td><td>{fmt_large(float_s)}</td></tr>
<tr><td>P/E (TTM)</td><td>{f"{info_d.get('trailingPE',0):.1f}x" if info_d.get('trailingPE') else 'N/A'}</td></tr>
<tr><td>Fwd P/E</td><td>{f"{info_d.get('forwardPE',0):.1f}x" if info_d.get('forwardPE') else 'N/A'}</td></tr>
</table>""",unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
        # Key metrics commentary
        if eps_g>20 and rev_g>15:
            st.markdown('<div class="signal-good">✅ Strong dual growth: EPS and Revenue both accelerating. High quality earnings confirm fundamentals.</div>',unsafe_allow_html=True)
        elif eps_g<0 or rev_g<0:
            st.markdown('<div class="signal-alert">⚠️ Declining EPS/Revenue — technical setup must be exceptional to justify trading against fundamental headwinds.</div>',unsafe_allow_html=True)
        if inst>70:
            st.markdown('<div class="signal-good" style="margin-top:6px">🏛️ High institutional ownership ({:.0f}%) — smart money committed. Supply controlled by long-term holders.</div>'.format(inst),unsafe_allow_html=True)

    st.divider()

    # Insider Activity
    st.markdown("### 🏢 Insider Activity")
    insider_data=get_insider_activity(ticker)
    sig_color={"Bullish":"#26a69a","Bearish":"#ef5350","Neutral":"#787b86"}.get(insider_data["net_signal"],"#787b86")
    ic1,ic2=st.columns([2,3])
    with ic1:
        st.markdown(f'<div class="card card-accent-blue"><strong>Net Signal: <span style="color:{sig_color}">{insider_data["net_signal"]}</span></strong><br><span style="color:#787b86;font-size:12px">{insider_data["summary"]}</span></div>',unsafe_allow_html=True)
    with ic2:
        if insider_data["transactions"]:
            st.dataframe(pd.DataFrame(insider_data["transactions"]),use_container_width=True,hide_index=True,height=180)
        else:
            st.info("Insider transaction data not available for this ticker.")
    st.divider()

    # MTF
    st.markdown("### 🕐 Multi-Timeframe Confirmation")
    align=mtf["alignment_score"]
    ac="#26a69a" if align>=65 else ("#f5c842" if align>=45 else "#ef5350")
    st.markdown(f"**Alignment:** <span style='color:{ac};font-size:22px;font-weight:700'>{align}/100</span> — {mtf['primary_bias']}",unsafe_allow_html=True)
    ma,mb,mc_col,md=st.columns(4)
    ma.metric("Bias",mtf["primary_bias"])
    mb.metric("Trigger",mtf["trigger"][:40]+"…" if len(mtf["trigger"])>40 else mtf["trigger"])
    mc_col.metric("Conflict",mtf["conflict_note"][:40]+"…" if len(mtf["conflict_note"])>40 else mtf["conflict_note"])
    md.metric("Score",f"{align}/100")
    with st.expander("📋 Full Timeframe Table",expanded=False):
        st.dataframe(pd.DataFrame(mtf["rows"]),use_container_width=True,hide_index=True)
    st.divider()

    # Options
    st.markdown("### 🎯 Options Setup")
    oa,ob,oc,od=st.columns(4)
    oa.metric("Recommendation",opt_vehicle)
    ob.metric("Next Expiry",opt_data.get("next_exp") or "N/A")
    oc.metric("IV (Median)",f"{opt_data['iv']:.1f}%" if opt_data.get("iv") else "N/A")
    od.metric("Event Risk",catalyst["event_risk"])
    st.info(f"**Reasoning:** {opt_reason}")
    if opt_data.get("error"): st.caption(f"ℹ️ {opt_data['error']}")
    st.divider()

    # AI Comments
    st.markdown("### 🤖 Expert AI Analysis")
    eff_prov=st.session_state.get("ai_provider","none")
    if not st.session_state.get("ai_enabled",True): eff_prov="none"
    if st.button("🤖 Generate Expert Analysis",key="gen_ai"):
        with st.spinner("Generating expert analysis..."):
            ai_payload={
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
                "vwap":a["vwap"],"days_to_earnings":catalyst["days_to_earnings"],"sector":a["sector"],
                "setup_quality_score":50,"setup_grade":"B","momentum_score":50,"momentum_class":"Moderate",
                "structure":"Unknown","phase":"Unknown","setup_type":"Unknown",
            }
            commentary,ai_warn=generate_ai_commentary(ai_payload,provider=eff_prov,
                                                        model=st.session_state.get("ai_model","openrouter/auto"))
        if ai_warn: st.warning(ai_warn)
        st.markdown('<div class="ai-box">',unsafe_allow_html=True)
        st.markdown(commentary); st.markdown('</div>',unsafe_allow_html=True)
    else:
        st.caption("Click the button above to generate comprehensive expert analysis.")
    st.divider()

    # Analysis Scores — with pillar explanations on click
    st.markdown("### 📐 Analysis Scores")
    scores=a["scores"]; all_reasons=a["all_reasons"]
    labels=["Fundamentals","Technicals","Risk","Plan","Entry/Exit","Mindset"]
    icons =["💼","📊","🛡️","📋","🎯","🧠"]
    cols6=st.columns(6)
    for i,(label,icon) in enumerate(zip(labels,icons)):
        sc=scores.get(label,3); clr=score_color(sc)
        cols6[i].markdown(
            f'<div class="card" style="text-align:center"><div style="font-size:20px">{icon}</div>'
            f'<div style="color:#787b86;font-size:11px;margin:4px 0">{label}</div>'
            f'<div style="color:{clr};font-size:22px;font-weight:700">{sc:.1f}</div>'
            f'</div>',unsafe_allow_html=True)
    overall_color=score_color(a["overall_score"])
    st.markdown(
        f'<div style="background:#1e222d;border:1px solid #2a2e3e;border-radius:8px;padding:12px 18px;margin-top:8px">'
        f'<span style="color:#787b86">Overall Score: </span>'
        f'<span style="color:{overall_color};font-size:20px;font-weight:700">{a["overall_score"]:.1f}/5</span>'
        f' &nbsp;|&nbsp; Trend: <strong>{a["trend_bias"]}</strong>'
        f' &nbsp;|&nbsp; Entry: <strong>${a["entry_low"]:.2f}–${a["entry_high"]:.2f}</strong>'
        f' &nbsp;|&nbsp; Stop: <strong style="color:#ef5350">${a["stop_loss"]:.2f}</strong>'
        f' &nbsp;|&nbsp; T1: <strong style="color:#26a69a">${a["target_1"]:.2f}</strong>'
        f' &nbsp;|&nbsp; R:R: <strong>{a["rr"]:.2f}x</strong>'
        f'</div>',unsafe_allow_html=True)

    st.markdown("#### 📖 Score Explanations — click to expand")
    for label,icon in zip(labels,icons):
        sc=scores.get(label,3); clr=score_color(sc); badge=score_badge(sc)
        reasons=all_reasons.get(label,{"positive":[],"negative":[]})
        with st.expander(f"{icon} **{label}** — {sc:.1f}/5  {badge}",expanded=False):
            pos=reasons.get("positive",[]); neg=reasons.get("negative",[])
            if pos:
                st.markdown("**✅ Positive factors:**")
                for r in pos: st.markdown(f"- {r}")
            if neg:
                st.markdown("**⚠️ Negative factors:**")
                for r in neg: st.markdown(f"- {r}")
            if not pos and not neg:
                st.markdown("*Score based on composite calculation — expand Advanced Analysis for deeper breakdown.*")
    st.divider()

    # News + Social (scrollable, 5 visible, 20+ items)
    nc1,nc2=st.columns([1,2])
    with nc1:
        st.markdown("### 📊 Sentiment")
        sc_clr="#26a69a" if sent_label=="Bullish" else ("#ef5350" if sent_label=="Bearish" else "#f5c842")
        st.progress(sent_score/100)
        st.markdown(f"<span style='color:{sc_clr};font-weight:700;font-size:16px'>{sent_label} — {sent_score}/100</span>",unsafe_allow_html=True)
        st.caption(f"Based on {len(news)} items across multiple sources")
        st.divider()
        st.markdown("### 💬 Social Comments")
        st.caption("Community signals from Reddit & market feeds")
        social=get_social_comments(ticker)
        if social:
            html='<div class="social-scroll">'
            for item in social[:25]:
                html+=(f'<div class="social-item">'
                       f'<span style="color:#787b86;font-size:10px">{item["platform"]} · {item["date"]}</span><br>'
                       f'{item["sentiment"]} <span style="font-size:12px;color:#d1d4dc">{item["text"]}</span>')
                if item.get("body"): html+=f'<br><span style="color:#787b86;font-size:11px">{item["body"][:120]}</span>'
                html+='</div>'
            html+='</div>'
            st.markdown(html,unsafe_allow_html=True)
        else:
            st.info("Social data loading…")

    with nc2:
        st.markdown(f"### 📰 Latest News ({len(news)} items)")
        html='<div class="news-scroll">'
        for item in news[:25]:
            t_low=item["title"].lower()
            bull_c=sum(1 for w in BULL_WORDS if w in t_low); bear_c=sum(1 for w in BEAR_WORDS if w in t_low)
            dot="🟢" if bull_c>bear_c else ("🔴" if bear_c>bull_c else "⚪")
            html+=(f'<div class="news-item">{dot} '
                   f'<a href="{item["link"]}" target="_blank" style="color:#7ab4ff;text-decoration:none;font-size:12px">{item["title"]}</a>'
                   f'<br><span style="color:#787b86;font-size:10px">{item.get("source","")} · {item.get("published","")}</span>'
                   f'</div>')
        html+='</div>'
        st.markdown(html,unsafe_allow_True=True,unsafe_allow_html=True)
    st.divider()

    # Save to Journal
    with st.expander("💾 Save Setup to Journal",expanded=False):
        j1,j2=st.columns(2)
        with j1:
            j_dir=st.selectbox("Direction",["Long","Short"],key="j_dir")
            j_at =st.selectbox("Asset Type",["Stock","Option","Spread"],key="j_at")
            j_th =st.text_area("Thesis",key="j_th",placeholder="Why are you taking this trade?")
            j_tf =st.selectbox("Timeframe",["1m","5m","15m","1h","4h","1d"],index=5,key="j_tf")
        with j2:
            j_res=st.selectbox("Result",["Open","Win","Loss","Breakeven"],key="j_res")
            j_pnl=st.number_input("P&L ($)",value=0.0,key="j_pnl")
            j_nt =st.text_area("Notes",key="j_nt"); j_mis=st.text_area("Mistakes",key="j_mis")
        if st.button("💾 Save",key="save_j"):
            save_trade({"date":datetime.now().strftime("%Y-%m-%d %H:%M"),"ticker":ticker,
                        "direction":j_dir,"asset_type":j_at,"entry":a["entry_high"],
                        "stop":a["stop_loss"],"target":a["target_1"],"size":"",
                        "thesis":j_th,"timeframe":j_tf,"alignment_score":mtf["alignment_score"],
                        "catalyst_state":catalyst["risk_label"],"options_setup":opt_vehicle,
                        "result":j_res,"pnl":j_pnl,"notes":j_nt,"mistakes":j_mis})
            st.success(f"✅ Saved: {ticker} {j_dir}")

# ══════════════════════════════════════════════════════════════
#  PAGE 2: ADVANCED ANALYSIS
# ══════════════════════════════════════════════════════════════
elif "🔬" in page:
    st.markdown("## 🔬 Advanced Analysis — Decision Engine")

    c1,c2,c3=st.columns([3,1,1])
    with c1:
        default_tick=st.session_state.get("ticker_input","AVGO")
        ticker=st.text_input("Ticker",value=default_tick,placeholder="e.g. AVGO, META",key="adv_ticker").strip().upper()
    with c2:
        interval=st.selectbox("Interval",["1m","5m","15m","1h","4h","1d","1wk"],index=5,key="adv_interval")
    with c3:
        st.markdown("<br>",unsafe_allow_html=True)
        st.button("▶ Load",use_container_width=True,key="adv_load")

    if not ticker: st.info("Enter a ticker symbol above."); st.stop()

    df,info,error,meta=get_stock_data(ticker,interval)
    if error or df is None: st.error(f"Could not load **{ticker}**: {error}"); st.stop()

    news=get_news(ticker); sent_score,sent_label=calculate_sentiment(news)
    a  =analyze(df,info or {},ticker,sent_score)
    catalyst=get_catalyst_data(ticker); opt_data=get_options_data(ticker)
    mtf=get_mtf_data(ticker)

    # Compute all advanced modules
    vol_intel=compute_volume_intelligence(df,a)
    vol_ctx  =compute_volatility_context(df,a)
    inst_pos =compute_institutional_positioning(a,df)
    market_st=compute_market_structure(df)
    mom      =compute_momentum_score(df,a,mtf)
    risk_prof=compute_risk_profile(a)
    setup_q  =compute_setup_quality(a,vol_intel,mom,market_st,catalyst,mtf,opt_data)
    trade_cls=classify_trade_setup(a,market_st,vol_intel,mom)
    exp_move =compute_expected_move(a,opt_data)
    risks    =compute_failure_risks(a,vol_intel,mom,market_st,catalyst,mtf,opt_data,meta)

    cur=float(df["Close"].iloc[-1])
    company_name=a.get("long_name",ticker)
    if company_name and company_name!=ticker:
        st.markdown(f"<span style='color:#787b86;font-size:13px'>{company_name} — Advanced Analysis</span>",unsafe_allow_html=True)

    # ── ROW 1: Top 4 score cards ──────────────────────────────
    r1a,r1b,r1c,r1d=st.columns(4)
    with r1a:
        gc=setup_q["grade_color"]
        st.markdown(f'<div class="adv-card"><h4>🏆 Setup Quality</h4>'
                    f'<span class="score-grade" style="color:{gc}">{setup_q["grade"]}</span>'
                    f' <span style="color:#787b86;font-size:16px">({setup_q["score"]}/100)</span>'
                    f'<br><span style="color:#787b86;font-size:11px">{setup_q["note"]}</span>'
                    f'</div>',unsafe_allow_html=True)
    with r1b:
        mc={"green":"#26a69a","gold":"#f5c842","red":"#ef5350"}.get(mom["mom_color"],"#f5c842")
        st.markdown(f'<div class="adv-card"><h4>⚡ Momentum Score</h4>'
                    f'<span class="score-grade" style="color:{mc}">{mom["score"]}</span>'
                    f'<span style="color:#787b86;font-size:14px">/100</span>'
                    f' <span style="color:{mc};font-size:13px;font-weight:600">{mom["classification"]}</span>'
                    f'<br><span style="color:#787b86;font-size:11px">{mom["note"][:100]}</span>'
                    f'</div>',unsafe_allow_html=True)
    with r1c:
        rc={"Low Risk":"#26a69a","Medium Risk":"#f5c842","High Risk":"#ef5350"}.get(risk_prof["risk_class"],"#f5c842")
        st.markdown(f'<div class="adv-card"><h4>🛡️ Risk Profile</h4>'
                    f'<span style="color:{rc};font-size:22px;font-weight:800">{risk_prof["risk_class"]}</span>'
                    f'<br><span style="color:#787b86;font-size:11px">Vol: {risk_prof["vol_daily"]:.2f}%/day · '
                    f'ATR: {risk_prof["atr_pct"]:.2f}% · MDD: {risk_prof["max_dd"]:.1f}%</span>'
                    f'<br><span style="color:#787b86;font-size:11px">{risk_prof["behavior"]}</span>'
                    f'</div>',unsafe_allow_html=True)
    with r1d:
        st.markdown(f'<div class="adv-card"><h4>📏 Expected Move</h4>'
                    f'<span style="color:#26a69a;font-size:16px;font-weight:700">▲ ${exp_move["exp_up_atr"]:.2f}</span><br>'
                    f'<span style="color:#ef5350;font-size:16px;font-weight:700">▼ ${exp_move["exp_down_atr"]:.2f}</span><br>'
                    f'<span style="color:#787b86;font-size:11px">ATR-based 1.5× range</span><br>'
                    f'<span style="color:#787b86;font-size:11px">{exp_move["note_iv"]}</span>'
                    f'</div>',unsafe_allow_html=True)
    st.markdown("")

    # ── ROW 2: Market Structure + Institutional ───────────────
    r2a,r2b=st.columns(2)
    with r2a:
        sc={"green":"#26a69a","gold":"#f5c842","red":"#ef5350"}.get(market_st["struct_color"],"#f5c842")
        st.markdown(f'<div class="adv-card"><h4>📈 Market Structure</h4>'
                    f'<span style="color:{sc};font-size:18px;font-weight:700">{market_st["structure"]}</span>'
                    f' &nbsp; <span style="color:#787b86;font-size:13px">Phase: {market_st["phase"]}</span><br>'
                    f'<span style="font-size:12px;color:#d1d4dc">'
                    f'HH: {"✅" if market_st["hh"] else "❌"} &nbsp; '
                    f'HL: {"✅" if market_st["hl"] else "❌"} &nbsp; '
                    f'LH: {"✅" if market_st["lh"] else "❌"} &nbsp; '
                    f'LL: {"✅" if market_st["ll"] else "❌"}</span>',unsafe_allow_html=True)
        if market_st["bos"]:
            st.markdown(f'<br><span style="font-size:12px">{market_st["bos"]}</span>',unsafe_allow_html=True)
        st.markdown(f'<br><span style="color:#787b86;font-size:12px">{market_st["note"]}</span></div>',unsafe_allow_html=True)
    with r2b:
        vc={"green":"#26a69a","gold":"#f5c842","red":"#ef5350"}.get(inst_pos["vwap_color"],"#f5c842")
        st.markdown(f'<div class="adv-card"><h4>🏦 Institutional / VWAP Positioning</h4>'
                    f'<span style="color:{vc};font-size:18px;font-weight:700">{inst_pos["vwap_bias"]}</span>'
                    f' <span style="color:#787b86">({inst_pos["vwap_dev"]:+.2f}%)</span><br>'
                    f'<span style="color:#787b86;font-size:12px">VWAP: ${inst_pos["vwap"]:.2f}</span><br>'
                    f'<span style="color:#d1d4dc;font-size:12px">{inst_pos["note"]}</span><br>'
                    f'<span style="color:#787b86;font-size:12px">{inst_pos["inst_note"]}</span>'
                    f'</div>',unsafe_allow_html=True)
    st.markdown("")

    # ── ROW 3: Volume Intelligence + Volatility ───────────────
    r3a,r3b=st.columns(2)
    with r3a:
        pc={"green":"#26a69a","gold":"#f5c842","red":"#ef5350"}.get(vol_intel["participation_color"],"#f5c842")
        spike_badge='<span style="background:#ef5350;color:#fff;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700">🚨 SPIKE</span>' if vol_intel["spike"] else ""
        st.markdown(f'<div class="adv-card"><h4>📊 Volume Intelligence</h4>'
                    f'RVOL: <span style="color:{pc};font-size:18px;font-weight:700">{vol_intel["rvol"]:.2f}x</span> {spike_badge}<br>'
                    f'Volume: <span style="color:#d1d4dc">{vol_intel["vol"]:,.0f}</span> &nbsp; '
                    f'Avg: <span style="color:#787b86">{vol_intel["vol_avg"]:,.0f}</span><br>'
                    f'<span style="color:{pc};font-weight:600">{vol_intel["participation"]}</span><br>'
                    f'Candle Bias: <span style="color:#d1d4dc">{vol_intel["accum_distrib"]}</span><br>'
                    f'<span style="color:#787b86;font-size:12px">{vol_intel["note"]}</span>'
                    f'</div>',unsafe_allow_html=True)
    with r3b:
        vc={"Low":"#26a69a","Medium":"#f5c842","High":"#ef5350"}.get(vol_ctx["vol_state"],"#f5c842")
        compress_badge='<span style="background:#2962ff;color:#fff;padding:2px 8px;border-radius:10px;font-size:11px">SQUEEZE</span>' if vol_ctx["compression"] else ""
        expand_badge='<span style="background:#ef5350;color:#fff;padding:2px 8px;border-radius:10px;font-size:11px">EXPANDING</span>' if vol_ctx["expansion"] else ""
        st.markdown(f'<div class="adv-card"><h4>🌪️ Volatility Context</h4>'
                    f'State: <span style="color:{vc};font-size:18px;font-weight:700">{vol_ctx["vol_state"]}</span> '
                    f'{compress_badge}{expand_badge}<br>'
                    f'ATR: <span style="color:#d1d4dc">${vol_ctx["atr"]:.2f}</span> '
                    f'({vol_ctx["atr_pct"]:.2f}%)<br>'
                    f'Today move used: <span style="color:{"#ef5350" if vol_ctx["atr_used_pct"]>80 else "#26a69a"}">'
                    f'{vol_ctx["atr_used_pct"]:.0f}% of ATR</span><br>'
                    f'Expected range: <span style="color:#26a69a">${vol_ctx["exp_up"]:.2f}</span> / '
                    f'<span style="color:#ef5350">${vol_ctx["exp_down"]:.2f}</span><br>'
                    f'<span style="color:#787b86;font-size:12px">{vol_ctx["note"]}</span>'
                    f'</div>',unsafe_allow_html=True)
    st.markdown("")

    # ── ROW 4: Setup Classification + What Could Go Wrong ─────
    r4a,r4b=st.columns(2)
    with r4a:
        setup_colors={"Breakout":"#26a69a","Pullback":"#2962ff","Reversal":"#f5c842",
                      "Range Trade":"#787b86","Trend Continuation":"#26a69a","No Valid Setup":"#ef5350"}
        sc=setup_colors.get(trade_cls["setup"],"#787b86")
        st.markdown(f'<div class="adv-card"><h4>🏷️ Setup Classification</h4>'
                    f'<span style="color:{sc};font-size:22px;font-weight:800">{trade_cls["setup"]}</span><br>'
                    f'<span style="color:#787b86;font-size:12px;line-height:1.6">{trade_cls["note"]}</span>',
                    unsafe_allow_html=True)
        # Expected move check
        st.markdown(f'<br><span style="font-size:12px;color:#d1d4dc">T1: {exp_move["t1_note"]}</span><br>'
                    f'<span style="font-size:12px;color:#d1d4dc">T2: {exp_move["t2_note"]}</span>'
                    f'</div>',unsafe_allow_html=True)
    with r4b:
        st.markdown('<div class="adv-card"><h4>⚠️ What Could Go Wrong</h4>',unsafe_allow_html=True)
        for label,note in risks:
            color="#ef5350" if "🔴" in label else ("#f5c842" if "🟡" in label else "#26a69a")
            st.markdown(f'<div style="margin-bottom:6px"><span style="color:{color};font-weight:700;font-size:12px">{label}</span>'
                        f'<br><span style="color:#787b86;font-size:11px">{note}</span></div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
    st.markdown("")

    # ── Setup Quality Breakdown ───────────────────────────────
    with st.expander("📋 Setup Quality Score Breakdown",expanded=False):
        for item in setup_q["breakdown"]:
            st.markdown(f"- {item}")

    st.divider()

    # ── Advanced AI Desk Notes ────────────────────────────────
    st.markdown("### 🤖 Advanced AI Desk Notes")
    eff_prov=st.session_state.get("ai_provider","none")
    if not st.session_state.get("ai_enabled",True): eff_prov="none"
    if st.button("🔬 Generate Advanced Desk Notes",key="gen_adv_ai"):
        top_risks_str="; ".join([f"{lbl}: {note[:60]}" for lbl,note in risks[:3]])
        with st.spinner("Running advanced analysis..."):
            adv_payload={
                "ticker":ticker,"trend_bias":a["trend_bias"],"rr":a["rr"],
                "setup_quality_score":setup_q["score"],"setup_grade":setup_q["grade"],
                "momentum_score":mom["score"],"momentum_class":mom["classification"],
                "structure":market_st["structure"],"phase":market_st["phase"],
                "rvol":vol_intel["rvol"],"vol_participation":vol_intel["participation"],
                "atr_pct":vol_ctx["atr_pct"],"vol_state":vol_ctx["vol_state"],
                "atr_used_pct":vol_ctx["atr_used_pct"],
                "vwap_bias":inst_pos["vwap_bias"],"vwap_dev":inst_pos["vwap_dev"],
                "setup_type":trade_cls["setup"],"top_risks":top_risks_str,
                "exp_up":exp_move["exp_up_atr"],"exp_down":exp_move["exp_down_atr"],
                "mtf_align":mtf["alignment_score"],
            }
            adv_commentary,adv_warn=generate_advanced_ai(adv_payload,provider=eff_prov,
                                                           model=st.session_state.get("ai_model","openrouter/auto"))
        if adv_warn: st.warning(adv_warn)
        st.markdown('<div class="ai-box">',unsafe_allow_html=True)
        st.markdown(adv_commentary); st.markdown('</div>',unsafe_allow_html=True)
    else:
        st.caption("Click to generate AI desk notes for this advanced analysis.")

# ══════════════════════════════════════════════════════════════
#  PAGE 3: PORTFOLIO
# ══════════════════════════════════════════════════════════════
elif "💼" in page:
    st.markdown("## 💼 Portfolio")
    if "portfolio" not in st.session_state:
        st.session_state["portfolio"]=DEFAULT_PORTFOLIO.copy()
    portfolio=st.session_state["portfolio"]

    total_invested=0.0; total_value=0.0; rows=[]; alloc_labels=[]; alloc_vals=[]; today_pnl=0.0
    for tick,pos in portfolio.items():
        cp,pp=get_portfolio_price(tick)
        if cp is None: cp,pp=pos["avg_cost"],pos["avg_cost"]
        if pp is None: pp=cp
        invested=pos["shares"]*pos["avg_cost"]; value=pos["shares"]*cp
        pnl=value-invested; pnl_pct=pnl/invested*100 if invested else 0
        today_chg=(cp-pp)*pos["shares"]; today_pnl+=today_chg
        to_target=(pos["target"]-cp)/cp*100 if pos["target"] and cp else 0
        total_invested+=invested; total_value+=value
        implied_stop=pos["avg_cost"]*0.93; at_risk=cp<implied_stop
        rows.append({"Ticker":tick,"Shares":pos["shares"],"Avg Cost":f"${pos['avg_cost']:.4f}",
                     "Price":f"${cp:.2f}","Value":f"${value:,.2f}",
                     "P&L":f"${pnl:+,.2f} ({pnl_pct:+.2f}%)","Today":f"${today_chg:+,.2f}",
                     "Target":f"${pos['target']:.2f} (+{to_target:.1f}%)","Currency":pos["currency"],
                     "⚠️":"🔴 NEAR STOP" if at_risk else "✅"})
        alloc_labels.append(tick); alloc_vals.append(value)

    total_pnl=total_value-total_invested; cash_cad=CASH_USD*CAD_RATE; grand_total=total_value+cash_cad
    pv_color="#26a69a" if total_pnl>=0 else "#ef5350"; td_color="#26a69a" if today_pnl>=0 else "#ef5350"
    goal_pct=min(grand_total/GOAL_LOW*100,100) if GOAL_LOW>0 else 0

    pv,td,at_col,gl_col=st.columns(4)
    pv.markdown(f'<div class="card card-accent-blue" style="text-align:center">'
                f'<div style="color:#787b86;font-size:11px;text-transform:uppercase">Portfolio Value (CAD)</div>'
                f'<div style="font-size:28px;font-weight:700;color:#d1d4dc">${total_value:,.2f}</div>'
                f'<div style="color:#787b86;font-size:11px">+ ${cash_cad:,.0f} cash = <strong>${grand_total:,.0f}</strong></div>'
                f'</div>',unsafe_allow_html=True)
    td.markdown(f'<div class="card" style="text-align:center">'
                f'<div style="color:#787b86;font-size:11px;text-transform:uppercase">Today\'s Return</div>'
                f'<div style="font-size:24px;font-weight:700;color:{td_color}">${today_pnl:+,.2f}</div>'
                f'</div>',unsafe_allow_html=True)
    at_col.markdown(f'<div class="card" style="text-align:center">'
                    f'<div style="color:#787b86;font-size:11px;text-transform:uppercase">All-Time Return</div>'
                    f'<div style="font-size:24px;font-weight:700;color:{pv_color}">${total_pnl:+,.2f}</div>'
                    f'<div style="color:{pv_color};font-size:13px">{total_pnl/total_invested*100 if total_invested else 0:+.2f}%</div>'
                    f'</div>',unsafe_allow_html=True)
    gl_col.markdown(f'<div class="card" style="text-align:center">'
                    f'<div style="color:#787b86;font-size:11px;text-transform:uppercase">Goal Progress</div>'
                    f'<div style="font-size:20px;font-weight:700;color:#2962ff">{goal_pct:.1f}%</div>'
                    f'<div style="color:#787b86;font-size:11px">${grand_total:,.0f} / ${GOAL_LOW:,}</div>'
                    f'</div>',unsafe_allow_html=True)
    bar_w=min(100,goal_pct)
    st.markdown(f'<div style="background:#1e222d;border-radius:6px;height:8px;margin:8px 0">'
                f'<div style="background:linear-gradient(90deg,#2962ff,#26a69a);width:{bar_w}%;height:8px;border-radius:6px"></div>'
                f'</div><div style="text-align:center;color:#787b86;font-size:11px">Goal: ${GOAL_LOW:,}–${GOAL_HIGH:,} CAD</div>',unsafe_allow_html=True)
    st.divider()

    ch_col,hd_col=st.columns([1,2])
    with ch_col:
        av_c=alloc_vals+[cash_cad]; al_l=alloc_labels+["Cash"]
        colors=["#2962ff","#26a69a","#f5c842","#ef5350","#b388ff","#ff9800","#787b86"]
        pie=go.Figure(go.Pie(labels=al_l,values=av_c,hole=0.55,
            marker=dict(colors=colors[:len(av_c)],line=dict(color="#131722",width=2)),
            textinfo="label+percent",textfont=dict(size=11,color="#d1d4dc")))
        pie.update_layout(template="plotly_dark",paper_bgcolor="#1e222d",plot_bgcolor="#1e222d",
            height=280,margin=dict(l=10,r=10,t=10,b=10),showlegend=False,
            annotations=[dict(text=f"${total_value:,.0f}",x=0.5,y=0.5,showarrow=False,
                              font=dict(size=13,color="#d1d4dc"),align="center")])
        st.plotly_chart(pie,use_container_width=True,config={"displayModeBar":False})
    with hd_col:
        if rows: st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    st.divider(); st.markdown("### 🚨 Position Alerts")
    alert_shown=False
    for tick,pos in portfolio.items():
        cp,_=get_portfolio_price(tick)
        if cp is None: cp=pos["avg_cost"]
        implied_stop=pos["avg_cost"]*0.93; pnl_pct=(cp-pos["avg_cost"])/pos["avg_cost"]*100
        target_pct=(pos["target"]-cp)/cp*100 if pos["target"] else 0
        if cp<implied_stop:
            st.markdown(f'<div class="signal-alert">🔴 <strong>{tick} STOP LOSS ALERT</strong> — ${cp:.2f} breached 7% stop (${implied_stop:.2f}). Review immediately.</div>',unsafe_allow_html=True); alert_shown=True
        elif pnl_pct<-5:
            st.markdown(f'<div class="signal-alert">🟡 <strong>{tick} Warning</strong> — Down {pnl_pct:.1f}%. Stop at ${implied_stop:.2f}.</div>',unsafe_allow_html=True); alert_shown=True
        elif pnl_pct>15:
            st.markdown(f'<div class="signal-good">✅ <strong>{tick}</strong> — Up {pnl_pct:.1f}%. {target_pct:.1f}% to target. Consider partial profit.</div>',unsafe_allow_html=True); alert_shown=True
    if not alert_shown:
        st.markdown('<div class="signal-good">✅ All positions within normal parameters.</div>',unsafe_allow_html=True)

    st.divider()
    with st.expander("➕ Manage Positions"):
        pa,pb,pc,pd_c,pe=st.columns(5)
        with pa: ntick=st.text_input("Ticker",key="ntick").strip().upper()
        with pb: nshar=st.number_input("Shares",min_value=0.0,key="nshar")
        with pc: navg =st.number_input("Avg Cost",min_value=0.0,key="navg")
        with pd_c: ncur=st.selectbox("Currency",["CAD","USD"],key="ncur")
        with pe: ntgt=st.number_input("Target",min_value=0.0,key="ntgt")
        if st.button("Save Position") and ntick:
            st.session_state["portfolio"][ntick]={"shares":nshar,"avg_cost":navg,"currency":ncur,"target":ntgt}
            st.success(f"Saved {ntick}"); st.rerun()
        rm=st.selectbox("Remove",["—"]+list(portfolio.keys()))
        if st.button("Remove") and rm!="—":
            del st.session_state["portfolio"][rm]; st.success(f"Removed {rm}"); st.rerun()
    st.divider()
    cx,cy=st.columns(2)
    cx.metric("USD Cash Reserve",f"${CASH_USD:,.2f}"); cy.metric("CAD Equivalent (×1.38)",f"${cash_cad:,.0f}")

# ══════════════════════════════════════════════════════════════
#  PAGE 4: WATCHLIST
# ══════════════════════════════════════════════════════════════
elif "👁️" in page:
    st.markdown("## 👁️ Watchlist Scanner")
    if "watchlist" not in st.session_state:
        st.session_state["watchlist"]=DEFAULT_WATCHLIST.copy()
    wa,wb=st.columns([3,1])
    with wa: add_t=st.text_input("Add ticker").strip().upper()
    with wb:
        st.markdown("<br>",unsafe_allow_html=True)
        if st.button("Add") and add_t and add_t not in st.session_state["watchlist"]:
            st.session_state["watchlist"].append(add_t); st.success(f"Added {add_t}"); st.rerun()
    rm_w=st.selectbox("Remove",["—"]+st.session_state["watchlist"])
    if st.button("Remove from watchlist") and rm_w!="—":
        st.session_state["watchlist"].remove(rm_w); st.rerun()
    st.divider(); st.markdown("### 📡 Live Scan")
    st.caption("Click ticker to open in Chart & Analysis.")
    wl_rows=[]
    for tick in st.session_state["watchlist"]:
        df_w,info_w,_,_=get_stock_data(tick,"1d")
        if df_w is None or df_w.empty: continue
        lat=df_w.iloc[-1]; prev=df_w.iloc[-2] if len(df_w)>1 else lat
        pw=float(lat["Close"]); chg_w=(pw-float(prev["Close"]))/float(prev["Close"])*100
        rsi_w=float(lat["RSI"]) if "RSI" in df_w.columns else 0
        sd_w =str(lat["SupertrendDir"]) if "SupertrendDir" in df_w.columns else "—"
        e20_w=float(lat["EMA20"]) if "EMA20" in df_w.columns else pw
        e50_w=float(lat["EMA50"]) if "EMA50" in df_w.columns else pw
        trend_w="▲ Bull" if pw>e20_w>e50_w else ("▼ Bear" if pw<e50_w else "→ Neutral")
        vol_w=float(lat["Volume"]); vol_avg_w=float(lat.get("Vol_EMA20",vol_w))
        vr_w=vol_w/vol_avg_w if vol_avg_w>0 else 1.0
        bull_pts=0
        if pw>e20_w: bull_pts+=1
        if pw>e50_w: bull_pts+=1
        if sd_w=="up": bull_pts+=2
        if 50<rsi_w<70: bull_pts+=1
        if vr_w>1.2: bull_pts+=1
        try:
            cat_w=get_catalyst_data(tick)
            dte_w=cat_w.get("days_to_earnings")
            sig="⚠️ EARNINGS" if dte_w is not None and dte_w<=5 else \
                ("🟢 BUY" if bull_pts>=5 else ("🟡 WATCH" if bull_pts>=3 else "🔴 AVOID"))
        except Exception:
            sig="🟢 BUY" if bull_pts>=5 else ("🟡 WATCH" if bull_pts>=3 else "🔴 AVOID")
        wl_rows.append({"ticker":tick,"price":pw,"chg":chg_w,"rsi":rsi_w,"trend":trend_w,
                         "st":sd_w.upper(),"signal":sig,"vr":vr_w})
    if wl_rows:
        hdr=st.columns([1,1,1,1,1,1,1,1])
        for i,h in enumerate(["Ticker","Price","Chg%","RSI","Trend","ST","Vol/Avg","Signal"]):
            hdr[i].markdown(f"<span style='color:#787b86;font-size:11px;font-weight:700'>{h}</span>",unsafe_allow_html=True)
        for row in wl_rows:
            c1,c2,c3,c4,c5,c6,c7,c8=st.columns([1,1,1,1,1,1,1,1])
            with c1:
                if st.button(row["ticker"],key=f"wl_{row['ticker']}"):
                    st.session_state["ticker_input"]=row["ticker"]
                    st.session_state["page_override"]="📈 Chart & Analysis"; st.rerun()
            c2.write(f"${row['price']:.2f}")
            chg_c="#26a69a" if row["chg"]>=0 else "#ef5350"
            c3.markdown(f"<span style='color:{chg_c}'>{row['chg']:+.2f}%</span>",unsafe_allow_html=True)
            rsi_c="#ef5350" if row["rsi"]>70 else ("#26a69a" if row["rsi"]<30 else "#d1d4dc")
            c4.markdown(f"<span style='color:{rsi_c}'>{row['rsi']:.0f}</span>",unsafe_allow_html=True)
            c5.write(row["trend"])
            st_c="#26a69a" if row["st"]=="UP" else "#ef5350"
            c6.markdown(f"<span style='color:{st_c};font-weight:700'>{row['st']}</span>",unsafe_allow_html=True)
            vr_c="#26a69a" if row["vr"]>1.3 else ("#ef5350" if row["vr"]<0.7 else "#d1d4dc")
            c7.markdown(f"<span style='color:{vr_c}'>{row['vr']:.2f}x</span>",unsafe_allow_html=True)
            c8.write(row["signal"])

# ══════════════════════════════════════════════════════════════
#  PAGE 5: JOURNAL
# ══════════════════════════════════════════════════════════════
elif "📓" in page:
    st.markdown("## 📓 Trade Journal & Performance")
    jdf=load_journal(); analytics=journal_analytics(jdf)
    if analytics:
        st.markdown("### 📊 Performance Summary")
        j1,j2,j3,j4=st.columns(4)
        j1.metric("Total Trades",analytics["total"]); j2.metric("Win Rate",f"{analytics['win_rate']:.1f}%")
        j3.metric("Total P&L",f"${analytics['total_pnl']:+,.2f}")
        j4.metric("Profit Factor",f"{analytics['profit_factor']:.2f}x" if analytics['profit_factor']!=float("inf") else "∞")
        jb,jc,jd,_=st.columns(4)
        jb.metric("Avg Win",f"${analytics['avg_win']:+,.2f}"); jc.metric("Avg Loss",f"${analytics['avg_loss']:+,.2f}")
        jd.metric("Expectancy",f"${analytics['expectancy']:+,.2f}")
        for ins in analytics.get("insights",[]): st.info(ins)
        if not jdf.empty and "pnl" in jdf.columns:
            pnl_s=pd.to_numeric(jdf["pnl"],errors="coerce").fillna(0)
            if pnl_s.abs().sum()>0:
                fig_p=go.Figure()
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
        fa,fb=st.columns(2)
        with fa:
            ft=st.text_input("Ticker",key="ft"); fd=st.selectbox("Direction",["Long","Short"],key="fd")
            fa_=st.selectbox("Asset",["Stock","Option","Spread"],key="fa_")
            fe=st.number_input("Entry ($)",value=0.0,key="fe"); fs=st.number_input("Stop ($)",value=0.0,key="fs")
            fg=st.number_input("Target ($)",value=0.0,key="fg"); fsz=st.number_input("Size",value=0,key="fsz")
        with fb:
            fth=st.text_area("Thesis",key="fth"); ftf=st.selectbox("Timeframe",["1m","5m","15m","1h","4h","1d"],index=5,key="ftf")
            faln=st.slider("Alignment",0,100,50,key="faln"); fcat=st.selectbox("Catalyst",["Low Risk","Moderate Risk","High Risk"],key="fcat")
            fopt=st.text_input("Options Setup",key="fopt"); fres=st.selectbox("Result",["Open","Win","Loss","Breakeven"],key="fres")
            fpnl=st.number_input("P&L ($)",value=0.0,key="fpnl"); fnt=st.text_area("Notes",key="fnt"); fmis=st.text_area("Mistakes",key="fmis")
        if st.button("💾 Save Trade",key="save_mt"):
            if ft:
                save_trade({"date":datetime.now().strftime("%Y-%m-%d %H:%M"),"ticker":ft.upper(),
                            "direction":fd,"asset_type":fa_,"entry":fe,"stop":fs,"target":fg,"size":fsz,
                            "thesis":fth,"timeframe":ftf,"alignment_score":faln,"catalyst_state":fcat,
                            "options_setup":fopt,"result":fres,"pnl":fpnl,"notes":fnt,"mistakes":fmis})
                st.success("Saved!"); st.rerun()
    st.divider(); st.markdown("### 📋 Trade Log")
    if jdf.empty: st.info("No trades yet.")
    else:
        st.dataframe(jdf,use_container_width=True,hide_index=True)
        st.download_button("⬇ Download CSV",data=jdf.to_csv(index=False),file_name="trade_journal.csv",mime="text/csv")
        if st.button("🗑 Delete Last Trade"):
            jdf.iloc[:-1].to_csv(JOURNAL_FILE,index=False); st.success("Deleted."); st.rerun()
