from analysis import graph, correlation
import pandas as pd

def fix_pd(csv_file):
    ETF = pd.read_csv(csv_file)
    ETF['observation_date'] = pd.to_datetime(ETF['observation_date'])
    ETF.set_index('observation_date', inplace=True)
    return ETF


# Get data
ETF = fix_pd('data/cleanedData/QQQ_quarterly.csv')
ETF = ETF['Close']
MACRO = fix_pd('data/cleanedData/master_macro_table.csv')

# Analysis

# What now? You have some valuabvle functions and can start crunching numbers. 
#   First, pick one sector
#   Two, measure correlations, lots of them
#   Three, graphs
#   Four time series
master_table = MACRO.merge(ETF, on='observation_date', how='left')
correlation(master_table, 'XLP')


MACRO_specific = MACRO['FEDFUNDS']
graph(MACRO_specific, ETF, "XLP", "FEDFUNDS")

