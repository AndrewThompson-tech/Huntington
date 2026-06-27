import pandas as pd

def normalize_macros(macro_df: pd.DataFrame) -> pd.DataFrame:
    # interpolate the dataset to be monthly
    MACRO_CONFIG = {
        # Growth
        "GDP": {
            "ticker": "GDP",
            "native_frequency": "quarterly",
            "target_frequency": "MS",
            "aggregation_method": None,
            "fill_method": "ffill"
        },

        "Industrial_Production": {
            "ticker": "INDPRO",
            "native_frequency": "monthly",
            "target_frequency": "MS",
            "aggregation_method": None,
            "fill_method": None
        },

        "Retail_Sales": {
            "ticker": "RSAFS",
            "native_frequency": "monthly",
            "target_frequency": "MS",
            "aggregation_method": None,
            "fill_method": None
        },

        "Industrial Production: Manufacturing Index": {
            "ticker": "IPMAN",
            "native_frequency": "monthly",
            "target_frequency": "MS",
            "aggregation_method": None,
            "fill_method": None
        },

        # Inflation
        "CPI": {
            "ticker": "CPIAUCSL",
            "native_frequency": "monthly",
            "target_frequency": "MS",
            "aggregation_method": None,
            "fill_method": None
        },

        "Core_CPI": {
            "ticker": "CPILFESL",
            "native_frequency": "monthly",
            "target_frequency": "MS",
            "aggregation_method": None,
            "fill_method": None
        },

        "PPI": {
            "ticker": "PPIACO",
            "native_frequency": "monthly",
            "target_frequency": "MS",
            "aggregation_method": None,
            "fill_method": None
        },

        # Rates
        "Fed_Funds_Rate": {
            "ticker": "FEDFUNDS",
            "native_frequency": "monthly",
            "target_frequency": "MS",
            "aggregation_method": None,
            "fill_method": None
        },

        "10Y_Treasury": {
            "ticker": "GS10",
            "native_frequency": "monthly",
            "target_frequency": "MS",
            "aggregation_method": None,
            "fill_method": None
        },

        "2Y_Treasury": {
            "ticker": "GS2",
            "native_frequency": "monthly",
            "target_frequency": "MS",
            "aggregation_method": None,
            "fill_method": None
        },

        "Real_10Y_Yield": {
            "ticker": "DFII10",
            "native_frequency": "daily",
            "target_frequency": "MS",
            "aggregation_method": "last",
            "fill_method": "ffill"
        },

        "Yield_Curve_10Y_2Y": {
            "ticker": "T10Y2Y",
            "native_frequency": "daily",
            "target_frequency": "MS",
            "aggregation_method": "last",
            "fill_method": "ffill"
        },

        # Labor
        "Unemployment_Rate": {
            "ticker": "UNRATE",
            "native_frequency": "monthly",
            "target_frequency": "MS",
            "aggregation_method": None,
            "fill_method": None
        },

        "Nonfarm_Payrolls": {
            "ticker": "PAYEMS",
            "native_frequency": "monthly",
            "target_frequency": "MS",
            "aggregation_method": None,
            "fill_method": None
        },

        # Liquidity
        "M2_Money_Supply": {
            "ticker": "M2SL",
            "native_frequency": "monthly",
            "target_frequency": "MS",
            "aggregation_method": None,
            "fill_method": None
        },

        "Financial_Conditions_Index": {
            "ticker": "NFCI",
            "native_frequency": "weekly",
            "target_frequency": "MS",
            "aggregation_method": "last",
            "fill_method": "ffill"
        },

        "Corporate_Bond_Spread_BAA10Y": {
            "ticker": "BAA10Y",
            "native_frequency": "daily",
            "target_frequency": "MS",
            "aggregation_method": "last",
            "fill_method": "ffill"
        },

        # Consumer
        "Consumer_Confidence": {
            "ticker": "UMCSENT",
            "native_frequency": "monthly",
            "target_frequency": "MS",
            "aggregation_method": None,
            "fill_method": None
        },

        "Personal_Consumption_Expenditures": {
            "ticker": "PCE",
            "native_frequency": "monthly",
            "target_frequency": "MS",
            "aggregation_method": None,
            "fill_method": None
        },

        # Market Sentiment
        "VIX": {
            "ticker": "VIXCLS",
            "native_frequency": "daily",
            "target_frequency": "MS",
            "aggregation_method": "last",
            "fill_method": "ffill"
        },

        # Housing
        "Housing_Starts": {
            "ticker": "HOUST",
            "native_frequency": "monthly",
            "target_frequency": "MS",
            "aggregation_method": None,
            "fill_method": None
        },

        "Case_Shiller_Home_Index": {
            "ticker": "CSUSHPINSA",
            "native_frequency": "monthly",
            "target_frequency": "MS",
            "aggregation_method": None,
            "fill_method": None
        },

        # Commodities
        "WTI_Oil": {
            "ticker": "DCOILWTICO",
            "native_frequency": "daily",
            "target_frequency": "MS",
            "aggregation_method": "last",
            "fill_method": "ffill"
        },

        "Copper": {
            "ticker": "PCOPPUSDM",
            "native_frequency": "monthly",
            "target_frequency": "MS",
            "aggregation_method": None,
            "fill_method": None
        },

        # FX
        "Trade_Weighted_USD_Index": {
            "ticker": "DTWEXBGS",
            "native_frequency": "daily",
            "target_frequency": "MS",
            "aggregation_method": "last",
            "fill_method": "ffill"
        }
    }
    
    # store transfromations for each macro
    transformed_series = []

    # apply macro transformations
    for macro_col in macro_df.columns:
        macro_series = macro_df[macro_col].sort_index()
        config = MACRO_CONFIG[macro_col]

        # pull data from macro config
        aggregation_method = config['aggregation_method']
        fill_method = config['fill_method']
        target_frequency = config['target_frequency']

        # aggregate data if necessary
        if aggregation_method == 'last':
            macro_series = macro_series.resample(target_frequency).last()
        
        elif aggregation_method == 'mean':
            macro_series = macro_series.resample(target_frequency).mean()

        # handle missing values is necessary
        if fill_method == 'ffill':
            macro_series = macro_series.ffill()

        elif fill_method == 'interpolate':
            macro_series = macro_series.interpolate()

        # append series to output
        macro_series = macro_series.rename(macro_col) # ensure column names remain the same
        transformed_series.append(macro_series)

    # combine the transformed macro series into cleaned dataframe
    macro_df_clean = pd.concat(transformed_series, axis=1)

    # convert index of dataframe to monthly 
    monthly_macro_df = _create_monthly_index(macro_df_clean)


    return monthly_macro_df


def _create_monthly_index(macro_df: pd.DataFrame) -> pd.DataFrame:
    # create monthly indexed dataframe
    monthly_index = pd.date_range(
        start=macro_df.index.min(),
        end=macro_df.index.max(),
        freq="MS"
    )

    # apply monthly index to macro dataset
    macro_df = macro_df.reindex(monthly_index)

    # forward fill any missing values that might occur
    macro_df = macro_df.ffill()

    return macro_df