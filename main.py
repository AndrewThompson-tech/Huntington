from data_cleanse import *
from linearRegression import linear_regression
import pandas as pd



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
        "shift": 3
    },
    "UNRATE": {
        "path": "data/raw_data/UNRATE.csv",
        "pipeline": ["read"],
        "shift": 3
    },
    "MCOILWTICO": {
        "path": "data/raw_data/MCOILWTICO.csv",
        "pipeline": ["read"],
        "shift": 3
    },
    "PCEPI": {
        "path": "data/raw_data/PCEPI.csv",
        "pipeline": ["read"],
        "shift":  3 
    },
     "FEDFUNDS": {
        "path": "data/raw_data/FEDFUNDS.csv",
        "pipeline": ["read"],
        "shift": 3
    }
}

master_table(TABLE_CONFIG, PROCESSING)

# Get data
ETF = fix_pd('data/raw_data/XLE_monthly.csv')

ETF = ETF['Close']
MACRO = fix_pd('monthly_master_macro_table.csv')

master_table = MACRO.merge(ETF, on='observation_date', how='left')
x = master_table[["GDP", "UNRATE", "MCOILWTICO", "PCEPI", "FEDFUNDS"]]
y = master_table["Close"]

linear_regression(x, y)

