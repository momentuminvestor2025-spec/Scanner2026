import math
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Nifty Total Market 750 Scanner", layout="wide")

CONSTITUENTS_URL = "https://niftyindices.com/IndexConstituent/ind_niftytotalmarket_list.csv"
DEFAULT_RS_BENCHMARK = "^CNX500"
USDINR_FALLBACK = 83.0
MIN_MARKET_CAP_USD_DEFAULT = 1_000_000_000

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 1.2rem; max-width: 95rem;}
    .metric-card {
        background: linear-gradient(180deg, #141b2d 0%, #0f172a 100%);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 25px rgba(2,6,23,0.22);
        min-height: 110px;
    }
    .metric-label {color: #94a3b8; font-size: 0.88rem; margin-bottom: 0.35rem;}
    .metric-value {color: #f8fafc; font-size: 1.7rem; font-weight: 700; line-height: 1.1;}
    .metric-sub {color: #cbd5e1; font-size: 0.9rem; margin-top: 0.35rem;}
    .hero {
        padding: 1.3rem 1.4rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #0f172a 0%, #172554 55%, #0f766e 100%);
        color: white;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1rem;
    }
    .hero h1 {font-size: 2rem; margin: 0 0 .3rem 0;}
    .hero p {margin: 0; color: rgba(255,255,255,0.86);}
    .section-title {font-size: 1.1rem; font-weight: 700; margin: .25rem 0 .75rem 0;}
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 16px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(ttl=60 * 60 * 12)
def load_constituents() -> pd.DataFrame:
    df = pd.read_csv(CONSTITUENTS_URL)
    df.columns = [c.strip() for c in df.columns]
    symbol_col = None
    for c in df.columns:
        cl = c.lower()
        if cl in {"symbol", "ticker", "isin code"}:
            symbol_col = c
            break
    if symbol_col is None:
        candidates = [c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()]
        if candidates:
            symbol_col = candidates[0]
        else:
            raise ValueError(f"Could not identify symbol column in constituent file: {df.columns.tolist()}")
    name_col = next((c for c in df.columns if c.lower() in {"company name", "company", "name"}), symbol_col)
    df = df.rename(columns={symbol_col: "Symbol", name_col: "Company"})
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    df["YahooSymbol"] = df["Symbol"].apply(lambda x: x if x.endswith(".NS") else f"{x}.NS")
    return df[[c for c in ["Symbol", "Company", "YahooSymbol"] if c in df.columns]].drop_duplicates()

@st.cache_data(ttl=60 * 60 * 12)
def get_usdinr() -> float:
    try:
        fx = yf.Ticker("INR=X").history(period="5d", interval="1d", auto_adjust=False)
        if not fx.empty:
            return float(fx["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return USDINR_FALLBACK

@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def download_price_data(symbols, period="18mo"):
    return yf.download(
        tickers=list(symbols),
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def get_market_caps(symbols):
    rows = []
    for sym in symbols:
        cap = np.nan
        try:
            tk = yf.Ticker(sym)
            fi = getattr(tk, "fast_info", {}) or {}
            cap = fi.get("market_cap", np.nan)
        except Exception:
            cap = np.nan
        rows.append({"YahooSymbol": sym, "marketCap": cap})
    return pd.DataFrame(rows)


def extract_ohlc(data, symbol):
    if data is None or len(data) == 0:
        return pd.DataFrame()
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if symbol in data.columns.get_level_values(0):
                df = data[symbol].copy()
            else:
                cols = [c for c in data.columns if c[0] == symbol]
                if not cols:
                    return pd.DataFrame()
                df = data.loc[:, cols].copy()
                df.columns = [c[1] for c in cols]
        else:
            df = data.copy()
    except Exception:
        return pd.DataFrame()
    needed = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[needed].copy()
    if "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    return df.dropna(subset=["Close"])


def rma(series, length):
    return series.ewm(alpha=1 / length, adjust=False).mean()


def compute_metrics(df, benchmark_df):
    if df.empty or len(df) < 220:
        return None
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    ema10 = close.ewm(span=10, adjust=False).mean().iloc[-1]
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    sma100 = close.rolling(100).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    price = close.iloc[-1]

    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr14 = rma(tr, 14).iloc[-1]
    atr_pct = (atr14 / price) * 100 if price and not pd.isna(atr14) else np.nan

    hh20 = high.rolling(20).max().iloc[-1]
    ll20 = low.rolling(20).min().iloc[-1]
    rng = hh20 - ll20
    range_pos_pct = ((price - ll20) / rng * 100) if rng and not pd.isna(rng) else np.nan

    lookbacks = {"1W": 5, "1M": 21, "3M": 63, "6M": 126}
    returns = {k: (price / close.iloc[-v-1] - 1) * 100 if len(close) > v else np.nan for k, v in lookbacks.items()}

    if benchmark_df is not None and not benchmark_df.empty:
        bclose = benchmark_df["Close"].astype(float).reindex(close.index).ffill().dropna()
        aligned = pd.concat([close, bclose], axis=1, keys=["stock", "bench"]).dropna()
        rel = aligned["stock"] / aligned["bench"]
        rs_ratio = {k: (rel.iloc[-1] / rel.iloc[-v-1] - 1) * 100 if len(rel) > v else np.nan for k, v in lookbacks.items()}
    else:
        rs_ratio = {k: np.nan for k in lookbacks}

    return {
        "Price": price,
        "EMA10": ema10,
        "SMA20": sma20,
        "SMA50": sma50,
        "SMA100": sma100,
        "SMA200": sma200,
        "TrendStack": bool(price >= ema10 >= sma20 >= sma50 >= sma100 >= sma200),
        "ATR14": atr14,
        "ATR_Pct": atr_pct,
        "RangePos20_Pct": range_pos_pct,
        **{f"Return_{k}": v for k, v in returns.items()},
        **{f"RSvsBench_{k}": v for k, v in rs_ratio.items()},
    }


def percentile_rank(series):
    return series.rank(pct=True, method="average") * 100


def metric_card(label, value, sub=""):
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{value}</div>
            <div class='metric-sub'>{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_summary_chart(df, rs_col):
    top = df[["Symbol", rs_col, "ATR_RSRank"]].dropna().sort_values(rs_col, ascending=False).head(15)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top[rs_col],
        y=top["Symbol"],
        orientation="h",
        marker=dict(color=top["ATR_RSRank"], colorscale="Viridis", showscale=True, colorbar=dict(title="ATR RS")),
        hovertemplate="%{y}<br>RS: %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(autorange="reversed"),
        xaxis_title="RS Rank",
        title="Top matches by RS rank",
        template="plotly_dark",
    )
    return fig


st.markdown(
    """
    <div class='hero'>
        <h1>Nifty Total Market 750 Scanner</h1>
        <p>Multi-tab workflow: Upload -> Scanner -> Results -> History.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

constituents = load_constituents()
usdinr = get_usdinr()

with st.sidebar:
    st.header("Scanner controls")
    rs_mode = st.radio("RS mode", ["Universe percentile rank", "Relative vs Nifty 500 benchmark"], index=0)
    rs_threshold = st.slider("RS threshold", 50, 100, 97)
    rs_periods = st.multiselect("RS periods (OR)", ["1W", "1M", "3M", "6M"], ["1W", "1M", "3M", "6M"])
    atr_rs_threshold = st.slider("ATR RS threshold", 1, 100, 50)
    range_threshold = st.slider("Price-to-20D range %", 0, 100, 50)
    market_cap_usd = st.number_input("Minimum market cap (USD)", min_value=0.0, value=float(MIN_MARKET_CAP_USD_DEFAULT), step=100000000.0, format="%.0f")
    scan_limit = st.slider("Symbols to process", 50, 750, 250, 50)
    show_debug = st.toggle("Show processed universe table", value=False)
    run_scan = st.button("Run scanner", type="primary", use_container_width=True)

row1 = st.columns(4)
with row1[0]:
    metric_card("Universe", f"{len(constituents)} stocks", "Nifty Total Market list")
with row1[1]:
    metric_card("USD/INR", f"{usdinr:,.2f}", "Used for market-cap translation")
with row1[2]:
    metric_card("Min mcap", f"${market_cap_usd/1e9:,.1f}B", f"₹{(market_cap_usd*usdinr)/1e7:,.0f} Cr")
with row1[3]:
    metric_card("Scan size", f"{scan_limit}", "Adjust upward on stronger hosting")

main_tab, universe_tab, notes_tab = st.tabs(["Scanner", "Universe", "Notes"])

with universe_tab:
    st.markdown("<div class='section-title'>Universe preview</div>", unsafe_allow_html=True)
    st.dataframe(constituents.head(25), use_container_width=True, hide_index=True)

with notes_tab:
    st.markdown("<div class='section-title'>Usage notes</div>", unsafe_allow_html=True)
    st.markdown(
        """
- RS is computed as a percentile rank across the processed universe, or versus Nifty 500 if selected.
- ATR RS uses ATR percent as a cross-sectional percentile rank.
- The app is optimized for interactive use, so 250-300 symbols is a good default for Streamlit hosting.
- Install with `pip install -r requirements.txt` and run using `streamlit run app.py`.
        """
    )

with main_tab:
    st.markdown("<div class='section-title'>Run the scan</div>", unsafe_allow_html=True)
    st.caption("Use the sidebar to set thresholds, then run the scanner.")

    if run_scan:
        universe = constituents.head(scan_limit).copy()
        symbols = universe["YahooSymbol"].tolist()

        with st.spinner("Downloading benchmark, price history, and market-cap data..."):
            benchmark_raw = yf.download(DEFAULT_RS_BENCHMARK, period="18mo", interval="1d", progress=False, auto_adjust=False)
            benchmark_df = extract_ohlc(benchmark_raw, DEFAULT_RS_BENCHMARK) if isinstance(benchmark_raw.columns, pd.MultiIndex) else benchmark_raw
            price_data = download_price_data(tuple(symbols), period="18mo")
            market_caps = get_market_caps(tuple(symbols))

        rows = []
        progress = st.progress(0, text="Processing symbols...")
        for i, sym in enumerate(symbols, start=1):
            df = extract_ohlc(price_data, sym)
            m = compute_metrics(df, benchmark_df)
            if m is not None:
                rows.append({"YahooSymbol": sym, **m})
            progress.progress(i / len(symbols), text=f"Processed {i}/{len(symbols)}")

        if not rows:
            st.error("No symbols produced enough data for analysis.")
            st.stop()

        result = pd.DataFrame(rows).merge(universe, on="YahooSymbol", how="left").merge(market_caps, on="YahooSymbol", how="left")

        for p in ["1W", "1M", "3M", "6M"]:
            result[f"RSRank_{p}"] = percentile_rank(result[f"Return_{p}"])
        result["ATR_RSRank"] = percentile_rank(result["ATR_Pct"])

        if rs_mode == "Universe percentile rank":
            rs_cols = [f"RSRank_{p}" for p in rs_periods]
        else:
            for p in ["1W", "1M", "3M", "6M"]:
                result[f"RSvsBenchRank_{p}"] = percentile_rank(result[f"RSvsBench_{p}"])
            rs_cols = [f"RSvsBenchRank_{p}" for p in rs_periods]

        result["RS_Pass"] = result[rs_cols].max(axis=1) >= rs_threshold if rs_cols else False
        result["ATR_RS_Pass"] = result["ATR_RSRank"] >= atr_rs_threshold
        result["Range_Pass"] = result["RangePos20_Pct"] >= range_threshold
        result["MCap_Pass"] = result["marketCap"] >= market_cap_usd
        result["All_Pass"] = result[["RS_Pass", "TrendStack", "ATR_RS_Pass", "Range_Pass", "MCap_Pass"]].all(axis=1)

        passed = result[result["All_Pass"]].copy()
        sort_col = rs_cols[0] if rs_cols else "ATR_RSRank"
        passed = passed.sort_values(by=[sort_col, "ATR_RSRank"], ascending=False)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            metric_card("Matches", f"{len(passed)}", f"of {len(result)} processed")
        with k2:
            metric_card("Hit rate", f"{(len(passed)/len(result)*100):.1f}%", "Pass ratio")
        with k3:
            top_rs = passed[sort_col].max() if len(passed) else float('nan')
            metric_card("Best RS", f"{top_rs:.1f}" if pd.notna(top_rs) else "—", sort_col)
        with k4:
            median_atr = passed["ATR_RSRank"].median() if len(passed) else float('nan')
            metric_card("Median ATR RS", f"{median_atr:.1f}" if pd.notna(median_atr) else "—", "Among matches")

        left, right = st.columns([1.25, 1])
        with left:
            st.markdown("<div class='section-title'>Matched stocks</div>", unsafe_allow_html=True)
            display_cols = [
                "Symbol", "Company", "Price", "EMA10", "SMA20", "SMA50", "SMA100", "SMA200",
                "RangePos20_Pct", "ATR_Pct", "ATR_RSRank", "marketCap"
            ] + rs_cols + ["TrendStack"]
            st.dataframe(
                passed[display_cols].rename(columns={"marketCap": "MarketCap_USD"}),
                use_container_width=True,
                hide_index=True,
            )
            csv = passed.to_csv(index=False).encode("utf-8")
            st.download_button("Download matches as CSV", data=csv, file_name="nifty_total_market_scan_results.csv", mime="text/csv")

        with right:
            st.markdown("<div class='section-title'>RS overview</div>", unsafe_allow_html=True)
            if len(passed):
                st.plotly_chart(build_summary_chart(passed, sort_col), use_container_width=True)
            else:
                st.info("No matches found for the current thresholds.")

        if show_debug:
            st.markdown("<div class='section-title'>Processed universe</div>", unsafe_allow_html=True)
            dbg_cols = ["Symbol", "Company", "Price", "RangePos20_Pct", "ATR_RSRank", "marketCap"] + rs_cols + ["TrendStack", "All_Pass"]
            st.dataframe(result[dbg_cols].rename(columns={"marketCap": "MarketCap_USD"}), use_container_width=True, hide_index=True)
    else:
        st.info("Set filters in the sidebar and click Run scanner to generate results.")
