import pandas as pd
import yfinance as yf

'''
Clean up CSV files and put them into a master set
'''

def read_quarterly(csv_file):
    # Make monthly->quarterly adjustments
    df = pd.read_csv(csv_file)

    if 'observation_date' not in df.columns:
        return "no observation_date found"
    
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df['date_diff'] = df['observation_date'].diff()
    median_diff = df['date_diff'].median()

    if median_diff.days > 80 and median_diff.days < 95:
        df = df.drop(columns=['date_diff'])
        return df
    
    df = df.drop(columns=['date_diff'])
    df.set_index('observation_date', inplace=True)
    df_quarterly = df.resample('QS').mean()

    return df_quarterly 

def master_table(data_paths):
    # data is a list of file paths to CSVs
    # putting them all togther into a combined csv with the same formatting
    data_csv = [None]*len(data_paths)
    for i, x in enumerate(data_paths):
        data_csv[i] = read_quarterly(x)

    master_table = data_csv[0]

    for df in data_csv[1:]:
        master_table = master_table.merge(df, on='observation_date', how='inner')

    master_table.to_csv('master_macro_table.csv', index=False)
    master_table.to_excel('master_macro_table.xlsx', index=False)

'''
    Example- 
    inflation = 'data/PCEPI.csv'
    gdp = 'data/GDP.csv'
    unemployment = 'data/UNRATE.csv'
    intrest_rates = 'data/FEDFUNDS.csv'
    oil_rates = 'data/MCOILWTICO.csv'

    data = inflation, gdp, unemployment, intrest_rates, oil_rates

    master_table(data)
'''
def get_ticker(ticker):
    '''
    Use yahoo finance API to get specific stock/ETF data
    '''

    # Cut off for master_table is 2025-07-01 
    df = yf.download(ticker, start="2000-01-01", end="2025-08-01",interval="1mo")

    df.reset_index(inplace=True)
    df.columns = ['observation_date', 'Close', 'High', 'Low', 'Open', 'Volume']
    df['observation_date'] = pd.to_datetime(df['observation_date'], errors='coerce')
    df = df.dropna(subset=['observation_date'])

    # Quarterly average
    df = df.set_index('observation_date')
    df_quarterly = df.resample('QS').last()  

    # reset index
    df_quarterly.reset_index(inplace=True)
    df_quarterly['observation_date'] = pd.to_datetime(df_quarterly['observation_date'])
    
    df_quarterly.to_csv(f'{ticker}_quarterly.csv')
    return df_quarterly