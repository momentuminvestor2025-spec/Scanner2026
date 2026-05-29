# Nifty Total Market 750 Streamlit Scanner

This app scans Nifty Total Market constituents using:
- Relative Strength threshold on 1W / 1M / 3M / 6M
- Trend stack: Price >= EMA10 >= SMA20 >= SMA50 >= SMA100 >= SMA200
- ATR relative-strength percentile
- Price-to-20-day range filter
- Market-cap filter

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- The app pulls the Nifty Total Market constituent CSV from Nifty Indices.
- NSE tickers are mapped to Yahoo Finance symbols by appending `.NS`.
- Full 750-symbol scans may be slow on low-resource hosting, so the UI includes a processing limit.


## Workflow

1. Upload the Nifty Total Market CSV into Supabase.
2. Run the scanner from the Streamlit Scanner tab.
3. Review results in the Results tab and optionally store them back into Supabase.

## Supabase tables

Use the `supabase_schema.sql` file to create the `nifty_total_market_universe` and `scanner_results` tables.

## Environment variables

Copy `.env.example` to `.env` and fill in `SUPABASE_URL` and `SUPABASE_KEY`.

## Streamlit Community Cloud Deployment

### Step 1: Create a new GitHub repository
1. Create a new public repo on GitHub (e.g., `nifty-scanner-streamlit`)
2. Push all files from this folder to the repo

### Step 2: Set up Supabase tables
1. Go to your Supabase project → SQL Editor
2. Copy contents of `supabase_schema.sql` and execute it
3. This creates `nifty_total_market_universe` and `scanner_results` tables

### Step 3: Add secrets to Streamlit Cloud
1. Go to your app on Streamlit Community Cloud
2. Click Settings → Secrets
3. Paste the following (the file `.streamlit/secrets.toml` is already configured):

```toml
SUPABASE_URL="postgresql://postgres.qlzgjloxdpwrrdkbwtkf:TXR0xujrBXymMTlx@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres"
SUPABASE_ANON_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJlZiI6InFsemdqbG94ZHB3cnJka2J3dGtmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzk4OTI5MzQsImV4cCI6MjA5NTQ2ODkzNH0.WLPRaeNFTLwfjkvqVSmF7MvMhsIq7WAiZ6z9QdVxltY"
```

4. Click Save and redeploy

### Step 4: Run the app
1. The app will load automatically
2. Go to Upload tab → Upload your CSV or use the official Nifty CSV
3. Go to Scanner tab → Set filters and click Run Scanner
4. View results in Results tab
