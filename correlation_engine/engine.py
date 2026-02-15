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
import sys
print(sys.path)
from .preprocessing import enforce_stationary
from .analyzer import chunkify, compute_lagged_correlations, aggregate_lags
from .config_generator import generate_json_config

def run_correlation_engine(master_df: pd.DataFrame, macro_columns: list, etf_columns: list, window_size: int, lags: int, generate_config=False):
    # ensure all data is stationary
    stationary_df, macro_transformations, etf_transformations = enforce_stationary(master_df, macro_columns, etf_columns)
    
    # create window chunks
    chunked_dfs = chunkify(stationary_df, window_size)

    # compute the best lag of each macro against each etf for each window
    all_window_lags = compute_lagged_correlations(chunked_dfs, macro_columns, etf_columns, lags)

    # determine the mode lag (best_lag) of each macro
    optimal_lags = aggregate_lags(all_window_lags)

    # only if the user wants to generate a json config 
    if generate_config:
        generate_json_config(optimal_lags)

    return optimal_lags
