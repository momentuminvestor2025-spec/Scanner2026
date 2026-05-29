import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
from supabase import create_client

st.set_page_config(page_title="Nifty Total Market 750 Scanner", layout="wide")

RS_THRESHOLD = 97
ATR_RS_THRESHOLD = 50
RANGE_THRESHOLD = 50
MCAP_THRESHOLD = 1_000_000_000
BENCHMARK = "^CNX500"
PERIOD = "18mo"


def create_supabase_client():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"])


def load_universe():
    supabase = create_supabase_client()
    resp = supabase.table("nifty_total_market_universe").select(
        "symbol,company,yahoo_symbol,sector,exchange,market_cap"
    ).execute()
    df = pd.DataFrame(resp.data or [])
    if df.empty:
        return df
    if "yahoo_symbol" not in df.columns:
        df["yahoo_symbol"] = (
            df["symbol"].astype(str).str.upper().str.strip()
            .apply(lambda x: x if x.endswith(".NS") else f"{x}.NS")
        )
    return df


def extract_ohlc(price_data, symbol):
    if isinstance(price_data.columns, pd.MultiIndex):
        if symbol in price_data.columns.get_level_values(0):
            df = price_data[symbol].copy()
        else:
            return pd.DataFrame()
    else:
        df = price_data.copy()
    if "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    return df.dropna(subset=["Close"])


def compute_scan_metrics(df):
    if df.empty or len(df) < 220:
        return None

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    price = close.iloc[-1]
    ema10 = close.ewm(span=10, adjust=False).mean().iloc[-1]
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    sma100 = close.rolling(100).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    atr14 = tr.rolling(14).mean().iloc[-1]
    atr_pct = (atr14 / price) * 100 if price else np.nan

    hh20 = high.rolling(20).max().iloc[-1]
    ll20 = low.rolling(20).min().iloc[-1]
    range_pos_20_pct = ((price - ll20) / (hh20 - ll20)) * 100 if hh20 > ll20 else np.nan

    def pct_return(lb):
        return ((price / close.iloc[-lb - 1]) - 1) * 100 if len(close) > lb else np.nan

    return {
        "price": price,
        "ema10": ema10,
        "sma20": sma20,
        "sma50": sma50,
        "sma100": sma100,
        "sma200": sma200,
        "trend_stack": bool(price >= ema10 >= sma20 >= sma50 >= sma100 >= sma200),
        "atr14": atr14,
        "atr_pct": atr_pct,
        "range_pos_20_pct": range_pos_20_pct,
        "return_1w": pct_return(5),
        "return_1m": pct_return(21),
        "return_3m": pct_return(63),
        "return_6m": pct_return(126),
    }


def build_scan_results(universe_df):
    if universe_df is None or universe_df.empty:
        return pd.DataFrame()

    dfu = universe_df.copy()
    if "market_cap" not in dfu.columns:
        dfu["market_cap"] = np.nan

    symbols = dfu["yahoo_symbol"].dropna().astype(str).tolist()
    benchmark = yf.download(
        BENCHMARK, period=PERIOD, interval="1d", auto_adjust=False, progress=False
    )
    price_data = yf.download(
        symbols,
        period=PERIOD,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    rows = []
    total = max(len(symbols), 1)
    prog = st.progress(0, text="Scanning symbols...")

    for i, sym in enumerate(symbols, start=1):
        stock_df = extract_ohlc(price_data, sym)
        m = compute_scan_metrics(stock_df)
        if m is not None:
            base = dfu[dfu["yahoo_symbol"] == sym].iloc[0].to_dict()
            rows.append({**base, **m})
        prog.progress(i / total, text=f"Processed {i}/{len(symbols)}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["rs_rank_1w"] = df["return_1w"].rank(pct=True, method="average") * 100
    df["rs_rank_1m"] = df["return_1m"].rank(pct=True, method="average") * 100
    df["rs_rank_3m"] = df["return_3m"].rank(pct=True, method="average") * 100
    df["rs_rank_6m"] = df["return_6m"].rank(pct=True, method="average") * 100
    df["atr_rs_rank"] = df["atr_pct"].rank(pct=True, method="average") * 100

    df["rs_pass"] = df[["rs_rank_1w", "rs_rank_1m", "rs_rank_3m", "rs_rank_6m"]].max(axis=1) >= RS_THRESHOLD
    df["atr_rs_pass"] = df["atr_rs_rank"] >= ATR_RS_THRESHOLD
    df["range_pass"] = df["range_pos_20_pct"] >= RANGE_THRESHOLD
    df["mcap_pass"] = df["market_cap"].fillna(0) >= MCAP_THRESHOLD
    df["all_pass"] = df[["rs_pass", "trend_stack", "atr_rs_pass", "range_pass", "mcap_pass"]].all(axis=1)

    return df.sort_values(by=["all_pass", "rs_rank_1w", "atr_rs_rank"], ascending=[False, False, False])


def build_charts(df):
    charts = {}
    if df is None or df.empty:
        return charts

    if "sector" in df.columns:
        sec = df["sector"].fillna("Unknown").value_counts().reset_index()
        sec.columns = ["Sector", "Count"]
        charts["sector"] = px.bar(sec.head(10), x="Sector", y="Count", title="Matches by sector")

    if "market_cap" in df.columns and "rs_rank_1w" in df.columns:
        scat = df[["market_cap", "rs_rank_1w", "symbol"]].dropna().copy()
        scat["market_cap"] = scat["market_cap"] / 1e9
        charts["scatter"] = px.scatter(
            scat,
            x="market_cap",
            y="rs_rank_1w",
            hover_data=["symbol"],
            title="Market cap vs RS 1W",
        )

    metric_cols = [
        c for c in [
            "rs_rank_1w",
            "rs_rank_1m",
            "rs_rank_3m",
            "rs_rank_6m",
            "atr_rs_rank",
            "range_pos_20_pct",
        ]
        if c in df.columns
    ]

    if metric_cols:
        m = df[metric_cols].melt(var_name="Metric", value_name="Value").dropna()
        m["Metric"] = m["Metric"].replace(
            {
                "rs_rank_1w": "RS 1W",
                "rs_rank_1m": "RS 1M",
                "rs_rank_3m": "RS 3M",
                "rs_rank_6m": "RS 6M",
                "atr_rs_rank": "ATR RS",
                "range_pos_20_pct": "Range %",
            }
        )
        charts["spread"] = px.box(m, x="Metric", y="Value", title="Metric spread")

    return charts


def main():
    st.title("Nifty Total Market 750 Scanner")
    tab1, tab2, tab3 = st.tabs(["Scanner", "Results", "History"])

    with tab1:
        st.subheader("Scanner")
        st.write(
            "Scan rules: RS >= 97 on any of 1W/1M/3M/6M, trend stack, ATR RS >= 50, range >= 50%, market cap >= $1B."
        )
        run = st.button("Run Scanner", type="primary")

        if run:
            universe_df = load_universe()
            if universe_df.empty:
                st.warning("No stocks found in Supabase universe table.")
                st.stop()

            results_df = build_scan_results(universe_df)
            if results_df.empty:
                st.warning("No scan results generated.")
                st.stop()

            passed_df = results_df[results_df["all_pass"]].copy()

            st.metric("Matches", len(passed_df))
            st.dataframe(passed_df, use_container_width=True, hide_index=True)

            if not passed_df.empty:
                supabase = create_supabase_client()
                run_id = pd.Timestamp.utcnow().strftime("run_%Y%m%d%H%M%S")
                passed_df["run_id"] = run_id
                supabase.table("scanner_results").insert(passed_df.to_dict("records")).execute()

                st.download_button(
                    "Download Results CSV",
                    data=passed_df.to_csv(index=False).encode("utf-8"),
                    file_name="nifty750_scanner_results.csv",
                    mime="text/csv",
                )

    with tab2:
        st.subheader("Results")
        supabase = create_supabase_client()
        resp = supabase.table("scanner_results").select("*").limit(500).execute()
        df = pd.DataFrame(resp.data or [])

        if df.empty:
            st.info("No stored results found.")
        else:
            if "created_at" in df.columns:
                df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
                df = df.sort_values("created_at", ascending=False)

            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Stored Results",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="stored_scanner_results.csv",
                mime="text/csv",
            )

            charts = build_charts(df)
            if charts:
                st.markdown("### Interactive charts")
                c1, c2 = st.columns(2)
                with c1:
                    if "sector" in charts:
                        st.plotly_chart(charts["sector"], use_container_width=True)
                    if "scatter" in charts:
                        st.plotly_chart(charts["scatter"], use_container_width=True)
                with c2:
                    if "spread" in charts:
                        st.plotly_chart(charts["spread"], use_container_width=True)

    with tab3:
        st.subheader("History")
        supabase = create_supabase_client()
        resp = supabase.table("scanner_results").select("run_id, created_at").limit(500).execute()
        hist = pd.DataFrame(resp.data or [])

        if hist.empty:
            st.info("No scan history yet.")
        else:
            if "created_at" in hist.columns:
                hist["created_at"] = pd.to_datetime(hist["created_at"], errors="coerce")
                hist = hist.sort_values("created_at", ascending=False)

            hist = hist.drop_duplicates()
            st.dataframe(hist, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
