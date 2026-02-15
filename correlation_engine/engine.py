import pandas as pd
'''
This file works as the main.py file for the correlation engine

Engine Inputs:
    1) Master_table --> Observation_date as index (interpolated monthly), all macro variables, all etf tickers
    2) Range of lags --> (-12 to 12) totaling 25 total lags
    3) List of all macro variables; will be important for separating the master_table
    4) List of all etf tickers; will be important for separating the master_table
    5) The window size for chunking, allows use to try different windows to test for differences can test (3 years, 5 years, 7 years, etc)
'''

def run_correlation_engine(master_df: pd.DataFrame, macro_columns: list, etf_columns: list, window_size: int, lags: int):
    """
    Simply calls all other fucntions from the package

    Ex:
        call split_macro_and_etf(); from data_laoder.py
        call create_time_windows(); from analyzer.py
        call compute_lagged_correlations(); from analyzer.py
        call aggregate_lags(); from analyzer.py
        
        if file doesn't already exist in ____filepath:
            call config_generator.py
    """
    pass
