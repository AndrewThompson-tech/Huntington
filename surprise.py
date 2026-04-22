from PCA import dynamic_pca
from data_cleanse import *
import pandas as pd
from sklearn.linear_model import LinearRegression

def expand_sep_to_monthly(sep):
    """
    Converts sparse SEP projections into monthly step-wise expectations.
    Each row holds until next available projection.
    """

    sep = sep.copy()
    sep = sep.sort_index()

    # Ensure monthly index covering full range
    full_index = pd.date_range(
        start=sep.index.min(),
        end=sep.index.max() + pd.DateOffset(years=1),
        freq="MS"
    )

    # Reindex + forward fill (KEY STEP)
    sep_monthly = sep.reindex(full_index).ffill()

    sep_monthly.index.name = "observation_date"

    return sep_monthly

def difference(merged, macro):
    expected = ((merged[macro+"L"] + merged[macro+"H"]) / 2) # Convert from percentage to decimal
    
    if macro=="GDP":
        actual = merged["GDP"].pct_change()  # GDP is in percentage points, convert to decimal
    elif macro == "UNRATE":
        actual = merged["UNRATE"]
    elif macro == "PCEPI":
        actual = merged["PCEPI"].pct_change()

    return actual - expected
    
    
def calculateSurprise(m_table):
    '''
    Calculate surprise from expected macro values. This uses SEP projections as expected and calculates the difference 
    from acutal. This is a pilot for what can be used with our own macor projections later. 
    '''
    sep = read_csv_standard('data/raw_data/June_2016_SEP.csv')
    sep.index = pd.to_datetime(sep.index)

    # Step 1: expand SEP correctly (NO interpolation)
    sep_monthly = expand_sep_to_monthly(sep)

    # Step 2: align indices
    m_table = m_table.copy()
    m_table.index = pd.to_datetime(m_table.index)

    # Step 3: merge expectations into macro table
    merged = m_table.merge(sep_monthly, on="observation_date", how="left")

    # Step 4: create SURPRISE variables


    gdp_diff = difference(merged, "GDP")
    unrate_diff = (difference(merged, "UNRATE"))
    pcepi_diff = (difference(merged, "PCEPI"))

    return gdp_diff, unrate_diff, pcepi_diff


if __name__ == "__main__":
    PROCESSING = {
        "read" : read_csv_standard,
        "quarterly" : read_quarterly,
        "MoM" : MoM,
        "interpolate_monthly" : interpolate_monthly,
        "YoY" : YoY,
        "enforce_stationary" : enforce_stationary,
        "log_diff" : log_diff,
        "diff" : diff
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
        "PCEPI": {
            "path": "data/raw_data/PCEPI.csv",
            "pipeline": ["read"], 
            "shift": 0
        }
    }

    TABLE_CONFIG_PROCESSED = {
        "GDP": {
            "path": "data/raw_data/GDP.csv",
            "pipeline": ["read", "interpolate_monthly", "log_diff"],
            "shift": 0
        },
        "UNRATE": {
            "path": "data/raw_data/UNRATE.csv",
            "pipeline": ["read", "log_diff"],
            "shift": 0
        },
        "PCEPI": {
            "path": "data/raw_data/PCEPI.csv",
            "pipeline": ["read", "log_diff"], 
            "shift": 0
        }
    }

    macros = master_table(TABLE_CONFIG_PROCESSED, PROCESSING, "master_macro_table.csv")
    macros_raw = master_table(TABLE_CONFIG, PROCESSING, "master_macro_table_raw.csv")

    etf = 'data/raw_data/ETFs/XLE_monthly.csv'
    etf = fix_pd(etf)
    y = etf["Close"].pct_change().dropna()
    print(y)

    macros_raw = macros_raw[204:236]
    gdp_diff, unrate_diff, pcepi_diff = calculateSurprise(macros_raw)

    # Train model
    X_train = macros[:204]
    Y_train = y[:204]
    Y_test = y[204:233]

    model = LinearRegression()
    model.fit(X_train, Y_train)
    print(model.coef_) # [7.62691617 0.05876542 3.35911731]]
    
    # Implement surprises. Change the model.coef_ based on calculateSurprise
    gdp_diff, unrate_diff, pcepi_diff = calculateSurprise(macros_raw)

# These macros are processed with log_diff
    X_test = macros[204:236].copy()

    surprise_df = pd.DataFrame({
        "GDP": gdp_diff,
        "UNRATE": unrate_diff,
        "PCEPI": pcepi_diff
    }).reindex(X_test.index)

    alpha = 0.5

    X_test[["GDP", "UNRATE", "PCEPI"]] += alpha * surprise_df
    X_test = X_test.dropna()

    # predict
    prediction = model.predict(X_test)


    # Visualize model here
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12,6))
    plt.plot(Y_test.index, Y_test, label="Actual ETF")
    plt.plot(Y_test.index, prediction, label="SEP-adjusted prediction")

    plt.legend()
    plt.title("ETF Prediction with Fed SEP Scenario Adjustment")
    plt.show()

