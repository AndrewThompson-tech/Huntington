import yfinance as yf

def get_etfs():
    '''
        Use yahoo finance API to get specific stock/ETF data
    '''

    # enforce global start and end dates for dataset
    start_date = '2006-01-01'
    end_date = '2026-01-01'

    gics_tickers = [
        "XLE", # Energy
        "XLF", # Financial Services
        "XLV", # Health Care
        "XLI", # Industrials
        "XLP", # Consumer Staples
        "XLY", # Consumer Discretionary
        "XLB", # Materials
        "XLU", # Utilities
        "XLK",  # Technology
        "VOX", # Communication Services == XLC
        "IYR" # Real Estate == XLRE
    ]

    master_df = yf.download(gics_tickers, start=start_date, end=end_date, interval="1mo")

    return master_df
