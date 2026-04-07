from data_cleanse import *
from linearRegression import LinearRegressionModel
from PCA import dynamic_pca
from correlation_engine.engine import run_correlation_engine
from correlation_engine.correlation import correlation
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
from timeseries import ARMAFamily


def create_linear_model( PROCESSING, TABLE_CONFIG, etf, use_lag=True, use_pca=True, corr_threshold=0.80, 
                        variance_explained=0.90, stability_threshold=0.50,
    ):
    valid_lag = []

    MACRO = master_table(TABLE_CONFIG, PROCESSING, "all_macros")
    ETF = fix_pd(etf)
    # print(ETF.head())
    ETF = ETF.pct_change()    
    
    m_table = MACRO.merge(ETF[['Close']], on='observation_date', how='left')
    m_table = m_table[:240]
    print(m_table)

    macros_for_corr = list(MACRO.columns)
    yearly_period, lags = 5, 12

    y = m_table["Close"]

    if use_lag:
        run_correlation_engine(m_table, macros_for_corr, ["Close"], yearly_period, lags, generate_config=True)
        m_table, valid_lag = apply_lag("optimal_lags.json", m_table, stability_threshold=stability_threshold)

    print(valid_lag)
    # Now remove Close (after lag engine is done)
    m_table = m_table.drop(columns=["Close"])

    if use_pca:
        MACRO_final = dynamic_pca(m_table, correlation_threshold=corr_threshold, variance_explained=variance_explained)
        MACRO_final.to_csv("pca_macros.csv")
    else:
        MACRO_final = m_table

    lr_model = LinearRegressionModel(MACRO_final, y, etf)
    # osl, anova = lr_model.linear_regression()

    return lr_model.linear_regression()

def create_time_series_model(PROCESSING, TABLE_CONFIG, etf, model_type="AUTO_ARIMA", forecast_periods=12):
    """
    Time series model workflow. Works with ARMA family of models.
    
    Parameters:
    - PROCESSING: dict of data processing functions
    - TABLE_CONFIG: dict of macro CSV paths and pipelines
    - etf: path to ETF CSV
    - model_type: str, one of ['AR', 'MA', 'ARIMA', 'ARIMAX', 'AUTO_ARIMA']
    - forecast_periods: int, number of future periods to forecast
    
    Returns:
    - forecast: pd.Series
    - metrics: dict with R² and directional accuracy
    - model_obj: fitted model object (ARMAFamily)
    """
    
    ETF = fix_pd(etf)
    ETF = ETF.pct_change().dropna()  # optional, aligns with your linear model

    MACRO = master_table(TABLE_CONFIG, PROCESSING, "all_macros")
    m_table = MACRO.merge(ETF[['Close']], on='observation_date', how='left')
    m_table = m_table[:240]  # align length
    print(m_table.head())
    
    y = m_table["Close"]
    X = m_table.drop(columns=["Close"])

    

    arma_model = ARMAFamily(X, y, etf)
    
    if model_type == "AUTO_ARIMA":
        forecast, fitted_model = arma_model.AUTO_ARIMA()
    elif model_type == "ARIMA":
        forecast, fitted_model = arma_model.ARIMA()
    elif model_type == "ARIMAX":
        forecast, fitted_model = arma_model.ARIMAX()
    elif model_type == "AR":
        forecast, fitted_model = arma_model.AR()
    elif model_type == "MA":
        forecast, fitted_model = arma_model.MA()
    else:
        raise ValueError(f"Unknown model_type {model_type}")
    
    results_df, dir_acc, r2_oos = arma_model.evaluate_forecast(forecast)
    metrics = {"R2": r2_oos, "Directional_Accuracy": dir_acc}
    
    # There is a bug when trying to save. Fix later.
    arma_model.plot_forecast(forecast, model_name=model_type, save=False)
    
    return forecast, metrics, arma_model




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
            "shift": 1
        },
        "MCOILWTICO": {
            "path": "data/raw_data/MCOILWTICO.csv",
            "pipeline": ["read", "log_diff"],
            "shift": 1
        },
        "UNRATE": {
            "path": "data/raw_data/UNRATE.csv",
            "pipeline": ["read", "log_diff"],
            "shift": 1
        },
     }

    
    etf = 'data/raw_data/ETFs/XLP_monthly.csv'

    # create_linear_model(
    #     PROCESSING,
    #     TABLE_CONFIG,
    #     etf,
    #     use_lag=True,
    #     use_pca=True,
    #     corr_threshold=0.80,
    #     variance_explained=0.90,
    #     stability_threshold=0.50,
    # )

    # I have things up and running. Now what i need to figure out is processing rules, what order I should give ARIMAX (will
    # probably need to run AUTO_ARIMA first). Then I need to implement the lag engine and all that good stuff. 
    # I am at a good stppping point for now.

    forecast, metrics, model_obj = create_time_series_model(
        PROCESSING,
        TABLE_CONFIG,
        etf,
        model_type="ARIMAX",
        forecast_periods=12
    )


