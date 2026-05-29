# Nifty Total Market 750 Scanner

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tabs

- Scanner
- Results
- History

## Notes

- Universe is read from Supabase table: `nifty_total_market_universe`
- Scan output is stored in `scanner_results`
- NSE symbols are mapped to Yahoo Finance using `.NS`
