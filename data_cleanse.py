import pandas as pd
import yfinance as yf

'''
Clean up CSV files and put them into a master set
'''
def read_csv_standard(csv_file):
    # Reads CSV file, making observation_date the index
    df = pd.read_csv(csv_file)
    
    if 'observation_date' not in df.columns:
        return "no observation_date found"
    
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df = df.sort_values('observation_date')
    df.set_index('observation_date', inplace=True)

    return df
        
def read_quarterly(df):
    # Takes df and extrapolates to quarterly average
    return df.resample('QE').mean()

def interpolate_monthly(df):
    # Takes quarterly measured data and interpolates it to monthly frequency, keeping a complete monthly date index.
    full_monthly_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq='MS'
    )
    # Reindex to full monthly dates
    df = df.reindex(full_monthly_index)
    # Interpolate missing values
    df_monthly = df.interpolate(method='linear')
    # Restore index name
    df_monthly.index.name = 'observation_date'
    return df_monthly

def MoM(df):
    # Calculate month over month % change
    df = df.pct_change() * 100
    df = df.dropna() # first row will be NaN
    return df

def YoY(df):
    """
    Takes data frame and finds % change from last year
    """
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) != 1:
        raise ValueError(f"Expected exactly one numeric column, found {list(numeric_cols)}")
    
    df_yoy = df.pct_change(periods=12) * 100
    df_yoy = df_yoy.dropna()
    
    return df_yoy

def fix_pd(csv_file):
    '''
    When you call a raw CSV file an index needs to be set
    
    :param csv_file: Description
    '''
    ETF = pd.read_csv(csv_file)
    ETF['observation_date'] = pd.to_datetime(ETF['observation_date'])
    ETF.set_index('observation_date', inplace=True)
    return ETF

def master_table(table_config, processing, name):
    dfs = []

    for series_name, cfg in table_config.items():
        df = None

        # Apply pipeline
        for step in cfg["pipeline"]:
            if df is None:
                df = processing[step](cfg["path"])
            else:
                df = processing[step](df)

        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) != 1:
            raise ValueError(
                f"{series_name} must have exactly one numeric data column, "
                f"found {list(numeric_cols)}"
            )

        df = df[numeric_cols]
        df.columns = [series_name]
        # Shift
        shift = cfg.get("shift", 0)
        if shift != 0:
            df[series_name] = df[series_name].shift(shift)

        dfs.append(df)

    # Merge all series on index
    master = dfs[0]
    for df in dfs[1:]:
        master = master.merge(df, left_index=True, right_index=True, how="inner")

    # Drop rows with NaNs created by shifts
    master = master.dropna()

    # Save
    master.to_csv(f"{name}.csv")

    return master

if __name__ == "__main__":
    PROCESSING = {
        "read" : read_csv_standard,
        "quarterly" : read_quarterly,
        "MoM" : MoM,
        "interpolate_monthly" : interpolate_monthly,
        "YoY" : YoY
    }

    TABLE_CONFIG = {
        "GDP": {
            "path": "data/raw_data/GDP.csv",
            "pipeline": ["read", "interpolate_monthly"],
            "shift": 0
        },
        "UNRATE": {
            "path": "data/raw_data/UNRATE.csv",
            "pipeline": ["read"],
            "shift": 0
        },
        "MCOILWTICO": {
            "path": "data/raw_data/MCOILWTICO.csv",
            "pipeline": ["read"],
            "shift": 0
        },
        "PCEPI": {
            "path": "data/raw_data/PCEPI.csv",
            "pipeline": ["read"],
            "shift":  0
        },
        "TOTALSA": {
            "path": "data/raw_data/TOTALSA.csv",
            "pipeline": ["read"],
            "shift": 0
        },
        "INDPRO": {
            "path": "data/raw_data/INDPRO.csv",
            "pipeline": ["read"],
            "shift": 0
        },
        "PPIENG": {
            "path": "data/raw_data/PPIENG.csv",
            "pipeline": ["read"],
            "shift": 0
        }
    }

    master_table(TABLE_CONFIG, PROCESSING, "all_macros")


