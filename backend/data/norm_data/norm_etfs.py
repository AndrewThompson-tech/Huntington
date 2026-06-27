import pandas as pd

def normalize_etfs(dirty_df: pd.DataFrame) -> pd.DataFrame:
    # drop the unnecessary columns
    dirty_df = dirty_df.drop(columns=['High', 'Low', 'Open', 'Volume'])

    # Columns are Multi-indexed by default; so each column is ['Close': 'ETF_Ticker']
    # flatten column names to only be the etf tickers
    cleaned_df = dirty_df['Close']

    # rename vanguard ticker columns to match the gics sector tickers
    cleaned_df = cleaned_df.rename(columns={'IYR': 'XLRE', 'VOX': 'XLC'})

    return cleaned_df

