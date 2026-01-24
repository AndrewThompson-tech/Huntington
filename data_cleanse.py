import pandas as pd

'''
Linear model analysing simple major macro measurements
'''

# file paths
inflation = 'data/PCEPI.csv'
gdp = 'data/GDP.csv'
unemployment = 'data/UNRATE.csv'
intrest_rates = 'data/FEDFUNDS.csv'
oil_rates = 'data/MCOILWTICO.csv'

def read_quarterly(csv_file):
    '''Make monthly-quarterly adjustments'''
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

# sloppy n lazy, fix later
inflation = read_quarterly(inflation)
gdp = read_quarterly(gdp)
unemployment = read_quarterly(unemployment)
intrest_rates = read_quarterly(intrest_rates)
oil_rates = read_quarterly(oil_rates)

data = inflation, gdp, unemployment, intrest_rates, oil_rates

for x in data:
    print(x.head())

master_table = data[0]

for df in data[1:]:
    master_table = master_table.merge(df, on='observation_date', how='inner')


print(master_table.head())


master_table.to_csv('master_macro_table.csv', index=False)
master_table.to_excel('master_macro_table.xlsx', index=False)
