from data_cleanse import *
from linearRegression import linear_regression
from PCA import dynamic_pca
from correlation_engine.engine import run_correlation_engine


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
    "FEDFUNDS": {
        "path": "data/raw_data/FEDFUNDS.csv",
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
        "shift": 0
    },
}

MACRO = master_table(TABLE_CONFIG, PROCESSING, "all_macros")
ETF = fix_pd('data/raw_data/XLE_monthly.csv')
master_table = MACRO.merge(ETF[['Close']], on='observation_date', how='left')

# macros_for_corr = list(MACRO.columns)
# window_size = 2
# lags = 12
# run_correlation_engine(master_table, macros_for_corr, ["Close"], window_size, lags, generate_config=True)

# # Apply leading lags now, pull from optimal_lags.json
# import json

# with open("optimal_lags.json", "r") as f:
#     optimal_lags = json.load(f)

# target = "Close"
# lags_for_target = optimal_lags[target]

# # Shift each column according to its optimal lag
# for col, lag in lags_for_target.items():
#     if col in master_table.columns:
#         master_table[col] = master_table[col].shift(lag)

# # Drop any rows with NaNs created by shifting
# master_table = master_table.dropna()


y = master_table["Close"]
master_table = master_table.drop(columns=["Close"])

MACRO_pca= dynamic_pca(master_table, correlation_threshold=0.8, variance_explained=0.95)
MACRO_pca.to_csv('pca_macros.csv')


linear_regression(MACRO_pca, y)

