from data_cleanse import *
from linearRegression import linear_regression
from PCA import dynamic_pca

PROCESSING = {
    "read" : read_csv_standard,
    "quarterly" : read_quarterly,
    "MoM" : MoM,
    "interpolate_monthly" : interpolate_monthly,
    "YoY" : YoY
}

TABLE_CONFIG = {
    "PCEPI": {
        "path": "data/raw_data/PCEPI.csv",
        "pipeline": ["read"],
        "shift": 0
    },
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
    }
}

MACRO = master_table(TABLE_CONFIG, PROCESSING, "all_macros")

MACRO_pca= dynamic_pca(MACRO, correlation_threshold=0.8, variance_explained=0.95)
MACRO_pca.to_csv('pca_macros.csv')

# # Get data
ETF = fix_pd('data/raw_data/XLE_monthly.csv')

master_table = MACRO_pca.merge(ETF, on='observation_date', how='left')


x = master_table.drop(columns=["Close", "High", "Low", "Open", "Volume"])
y = master_table["Close"]

linear_regression(x, y)

