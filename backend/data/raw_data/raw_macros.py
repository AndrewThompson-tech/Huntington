from fredapi import Fred
from dotenv import load_dotenv
import pandas as pd
import requests
import os

def get_macros():
    '''
        Retrieves a broad set of macroeconomic variables from the FRED API.

        The dataset intentionally includes correlated indicators across growth,
        inflation, rates, and liquidity regimes. Downstream, PCA is applied to
        reduce dimensionality and extract latent macroeconomic factors --> 
        mitigating multicollinearity.
    '''
    load_dotenv()
    fred = Fred(os.getenv("FRED_API_KEY"))

    all_macros = {
        # Growth
        "GDP": "GDP",
        "Industrial_Production": "INDPRO",
        "Retail_Sales": "RSAFS",
        "Industrial Production: Manufacturing Index": "IPMAN",

        # Inflation
        "CPI": "CPIAUCSL",
        "Core_CPI": "CPILFESL",
        "PPI": "PPIACO",

        # Rates
        "Fed_Funds_Rate": "FEDFUNDS",
        "10Y_Treasury": "GS10",
        "2Y_Treasury": "GS2",
        "Real_10Y_Yield": "DFII10",
        "Yield_Curve_10Y_2Y": "T10Y2Y",

        # Labor
        "Unemployment_Rate": "UNRATE",
        "Nonfarm_Payrolls": "PAYEMS",

        # Liquidity
        "M2_Money_Supply": "M2SL",
        "Financial_Conditions_Index": "NFCI",
        "Corporate_Bond_Spread_BAA10Y": "BAA10Y",

        # Consumer
        "Consumer_Confidence": "UMCSENT",
        "Personal_Consumption_Expenditures": "PCE",

        # Market Sentiment
        "VIX": "VIXCLS",

        # Housing
        "Housing_Starts": "HOUST",
        "Case_Shiller_Home_Index": "CSUSHPINSA",

        # Commodities
        "WTI_Oil": "DCOILWTICO",
        "Copper": "PCOPPUSDM",

        # FX
        "Trade_Weighted_USD_Index": "DTWEXBGS"
    }

    # enforce global start and end dates for dataset
    start_date = '2006-01-01'
    end_date = '2026-01-01'

    all_macro_series = []
    master_df = pd.DataFrame()

    for name, ticker in all_macros.items():
        try:
            # retrieve data from fred api
            macro = fred.get_series(ticker, observation_start=start_date, observation_end=end_date)

            # add current macro to macro list
            all_macro_series.append(macro.rename(name))
        
        except requests.exceptions.HTTPError as e:
            msg = str(e)

            # specify api key error
            if "401" in msg or "403" in msg:
                return None, "INVALID_API_KEY_OR_UNAUTHORIZED"

            # specify invalid series error
            if "400" in msg or "404" in msg:
                return None, "INVALID_SERIES_ID"

            return None, "HTTP_ERROR"

        except Exception as e:
            return None, f"UNKNOWN_ERROR: {str(e)}"
        
    # append all macro series to master dataset 
    master_df = pd.concat(all_macro_series, axis=1)

    return master_df

