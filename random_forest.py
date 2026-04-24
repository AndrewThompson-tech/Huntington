import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fredapi import Fred
from dotenv import load_dotenv
from correlation_engine import run_correlation_engine

class FeatureEngineer():
    MACRO_PATH = "data/raw_data/macros"
    ETF_PATH = "data/raw_data/ETFs"
    
    All_MACROS = {
            # Growth
            "GDP": "GDP",
            "Industrial_Production": "INDPRO",
            "Retail_Sales": "RSAFS",
            "Growth_Proxy": "INDPRO",  # ISM PMI

            # Inflation
            "CPI": "CPIAUCSL",
            "Core_CPI": "CPILFESL",
            "PPI": "PPIACO",

            # Rates
            "Fed_Funds_Rate": "FEDFUNDS",
            "10Y_Treasury": "GS10",
            "2Y_Treasury": "GS2",

            # Labor
            "Unemployment": "UNRATE",
            "Nonfarm_Payrolls": "PAYEMS",

            # Liquidity
            "M2": "M2SL",
            "Financial_Conditions": "NFCI",  # Chicago Fed

            # Consumer
            "Consumer_Confidence": "UMCSENT",
            "PCE": "PCE",

            # Commodities (FRED versions)
            "Oil_WTI": "DCOILWTICO",
            "Copper": "PCOPPUSDM"
        }
    
    MACROS_LIST = [
        "GDP",
        "Industrial_Production",
        "Retail_Sales",
        "Growth_Proxy",
        "CPI",
        "Core_CPI",
        "PPI",
        "Fed_Funds_Rate",
        "10Y_Treasury",
        "2Y_Treasury",
        "Unemployment",
        "Nonfarm_Payrolls",
        "M2",
        "Financial_Conditions",
        "Consumer_Confidence",
        "PCE",
        "Oil_WTI",
        "Copper"
    ]
    
    def __init__(self, etf_ticker):
        load_dotenv()
        self.fred = Fred(os.getenv("FRED_API_KEY"))
        os.makedirs(self.MACRO_PATH, exist_ok=True)
        self.etf_ticker = etf_ticker
        self.start_date = '2000-01-01'

    def load_data(self):
        macro_dfs = []

        # load or fetch macro data
        for name, macro_ticker in self.All_MACROS.items():
            file_path = os.path.join(self.MACRO_PATH, f"{name}.csv")

            if os.path.exists(file_path):
                # print(f"{name} file path exists")
                master_df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
            else:
                # print(f"{name} file path does not exist")
                data = self.fred.get_series(macro_ticker, observation_start=self.start_date)

                master_df = pd.DataFrame(data, columns=[name])
                master_df.index.name = "Date"

                master_df.to_csv(file_path)

            # clean each macro before merging
            master_df = master_df.sort_index()
            master_df = master_df[~master_df.index.duplicated(keep="first")]
            master_df = master_df.resample("MS").mean()

            macro_dfs.append(master_df)

        # merge the macro data
        # print("merging all macros together")
        macro_df = pd.concat(macro_dfs, axis=1).sort_index()

        # enforce MONTH START consistency
        macro_df = macro_df.resample("MS").mean()

        # enforce full MS timeline (NOT ME)
        full_index = pd.date_range(
            start=macro_df.index.min(),
            end=macro_df.index.max(),
            freq="MS"
        )
        macro_df = macro_df.reindex(full_index)

        macro_df = macro_df.interpolate(method="time")
        macro_df = macro_df.ffill().bfill()
        
        # load the etf
        # print("loading etf file")
        etf_file = os.path.join(
            self.ETF_PATH, f"{self.etf_ticker.upper()}_monthly.csv"
        )

        if not os.path.exists(etf_file):
            raise FileNotFoundError(f"{etf_file} not found")

        etf_df = pd.read_csv(etf_file, parse_dates=["observation_date"], index_col="observation_date")

        etf_df = etf_df[["Close"]].rename(
            columns={"Close": self.etf_ticker.upper()}
        )

        # merge etf + macros
        master_df = pd.concat([etf_df, macro_df], axis=1).sort_index()

        # Fill lower-frequency macro gaps (GDP, etc.)
        master_df = master_df.ffill()

        # Drop remaining NaNs
        master_df = master_df.dropna()

        # Add derived features
        master_df["Yield_Spread"] = master_df["10Y_Treasury"] - master_df["2Y_Treasury"]

        return master_df

    def apply_lags(self, master_df: pd.DataFrame):
        etf_list = [self.etf_ticker]

        # get optimal lags
        optimal_lags = run_correlation_engine(
            master_df=master_df,
            macro_columns=self.MACROS_LIST,
            etf_columns=etf_list,
            window_size=4,
            lags=12
        )

        # extract lags from dict
        lag_dict = {}
        for macro, values in optimal_lags[self.etf_ticker].items():
            lag_dict[macro] = values["lag"]
        
        # apply lags
        df_lagged = master_df.copy()

        for macro, lag in lag_dict.items():
            if macro in df_lagged.columns:
                df_lagged[macro] = df_lagged[macro].shift(lag)

        df_lagged = df_lagged.dropna()

        return df_lagged

    def create_target(self, lagged_df: pd.DataFrame):
        etf_col = self.etf_ticker

        lagged_df["Target"] = np.log(
            lagged_df[etf_col].shift(-1) / lagged_df[etf_col]
        )

        lagged_df = lagged_df.dropna(subset=["Target"])

        return lagged_df

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.tree import plot_tree

class RandomForestModel():
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        max_features="sqrt",
        n_splits=15
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_splits = n_splits

        self.models = []
        self.metrics = []
        self.final_model = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None

    def run_random_forest(self, df: pd.DataFrame):
        """
        df format:
        [ETF, macro features..., Target]
        """

        # Split features and target
        X = df.iloc[:, 1:-1]
        y = df["Target"]

        self.feature_names = X.columns.tolist()

        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = self._train_model(X_train, y_train)

            oob, mse, r2 = self._evaluate_model(model, X_test, y_test)

            self.models.append(model)
            self.metrics.append({
                "train_start": str(X_train.index[0]),
                "train_end": str(X_train.index[-1]),
                "test_start": str(X_test.index[0]),
                "test_end": str(X_test.index[-1]),
                "oob_score": oob,
                "mse": mse,
                "r2": r2
            })

        # store final model
        self.final_model = self.models[-1]
        self.X_test = X_test
        self.y_test = y_test

        return self

    def _train_model(self, X_train, y_train):
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            random_state=42,
            oob_score=True
        )

        model.fit(X_train, y_train)
        
        return model
    
    def _evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return model.oob_score_, mse, r2

    def get_metrics(self):
        return self.metrics
    
    def predict(self, X: pd.DataFrame):
        if self.final_model is None:
            raise ValueError("Model not trained yet.")

        # enforce exact training schema
        X = X.reindex(columns=self.feature_names)

        return self.final_model.predict(X)

    def feature_importance_gini(self):
        if self.final_model is None:
            raise ValueError("Model not trained yet.")

        importances = self.final_model.feature_importances_
        sorted_idx = np.argsort(importances)

        plt.barh(
            np.arange(len(sorted_idx)),
            importances[sorted_idx],
            align="center"
        )

        plt.yticks(
            np.arange(len(sorted_idx)),
            np.array(self.feature_names)[sorted_idx]
        )

        plt.title("Gini Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.show()

    def feature_importance_permutation(self):
        if self.final_model is None:
            raise ValueError("Model not trained yet.")

        results = permutation_importance(
            self.final_model,
            self.X_test,
            self.y_test,
            n_repeats=10,
            random_state=0,
            n_jobs=-1
        )

        sorted_idx = np.argsort(results.importances_mean)

        plt.barh(
            np.arange(len(sorted_idx)),
            results.importances_mean[sorted_idx],
            xerr=results.importances_std[sorted_idx],
            align='center'
        )

        plt.yticks(
            np.arange(len(sorted_idx)),
            np.array(self.feature_names)[sorted_idx]
        )

        plt.title("Permutation Feature Importance")
        plt.xlabel("Mean Decrease in Accuracy")
        plt.show()

class ScenarioEngine():
    SCENARIOS = {
        "growth_boom": {
            "GDP": 0.04,
            "Industrial_Production": 0.03,
            "Retail_Sales": 0.05,
            "Unemployment": 0.03,
            "CPI": 0.025,
            "Fed_Funds_Rate": 0.045
        },

        "recession_shock": {
            "GDP": -0.03,
            "Industrial_Production": -0.04,
            "Retail_Sales": -0.05,
            "Unemployment": 0.07,
            "CPI": 0.015,
            "Fed_Funds_Rate": 0.02
        },

        "inflation_spike": {
            "GDP": 0.01,
            "CPI": 0.06,
            "Core_CPI": 0.055,
            "PPI": 0.07,
            "Fed_Funds_Rate": 0.065,
            "Consumer_Confidence": -0.02
        },

        "rate_hike_cycle": {
            "GDP": 0.015,
            "CPI": 0.03,
            "Fed_Funds_Rate": 0.07,
            "10Y_Treasury": 0.06,
            "2Y_Treasury": 0.055,
            "Financial_Conditions": 1.2
        },

        "liquidity_flood": {
            "M2": 0.08,
            "Fed_Funds_Rate": 0.01,
            "Financial_Conditions": -0.5,
            "GDP": 0.03,
            "Retail_Sales": 0.04
        },

        "tech_boom_environment": {
            "GDP": 0.035,
            "Consumer_Confidence": 0.08,
            "M2": 0.04,
            "CPI": 0.02,
            "Industrial_Production": 0.02,
            "Copper": 0.03
        },

        "stagflation": {
            "GDP": 0.0,
            "CPI": 0.07,
            "Unemployment": 0.06,
            "Fed_Funds_Rate": 0.06,
            "Retail_Sales": -0.02
        },

        "oil_shock": {
            "Oil_WTI": 0.25,
            "CPI": 0.05,
            "GDP": -0.01,
            "Consumer_Confidence": -0.03,
            "Retail_Sales": -0.02
        },

        "deflation_risk": {
            "CPI": -0.01,
            "Core_CPI": -0.005,
            "GDP": 0.01,
            "Unemployment": 0.05,
            "Fed_Funds_Rate": 0.01
        },

        "balanced_slow_growth": {
            "GDP": 0.02,
            "CPI": 0.02,
            "Unemployment": 0.045,
            "Fed_Funds_Rate": 0.035,
            "Retail_Sales": 0.015
        }
    }

    def __init__(self, model, base_df: pd.DataFrame):
        self.model = model
        self.base_df = base_df.copy()

        self.baseline_predictions = None
        self.scenario_predictions = None
    
    def run_baseline(self):
        X = self.base_df[self.model.feature_names]  # macros only

        preds = self.model.predict(X)

        self.baseline_predictions = pd.Series(
            preds,
            index=self.base_df.index,
            name="Baseline"
        )

        return self.baseline_predictions

    def run_predefined_scenario(self, scenario_name: str):
        df = self.base_df.copy()

        if scenario_name not in self.SCENARIOS:
            raise ValueError(f"Scenario '{scenario_name}' not found")

        scenario_dict = self.SCENARIOS[scenario_name]

        # apply realistic shocks instead of overwriting values
        df = self._apply_shocks(df, scenario_dict)

        # align feature space exactly to training
        X = df[self.model.feature_names]

        preds = self.model.predict(X)

        self.scenario_predictions = pd.Series(
            preds,
            index=df.index,
            name=f"{scenario_name}_scenario"
        )

        return self.scenario_predictions    

    def run_custom_scenario(self, user_inputs: dict):
        """
            user_inputs format (from Streamlit):
            {
                "GDP": value,
                "CPI": value,
                ...
            }
        """

        df = self.base_df.copy()

        df = self._apply_shocks(df, user_inputs)

        X = df[self.model.feature_names]

        preds = self.model.predict(X)

        return pd.Series(
            preds,
            index=df.index,
            name="Custom_Scenario"
        )
    
    def _apply_shocks(self, df, scenario_dict):
        df = df.copy()

        for macro, shock in scenario_dict.items():
            if macro not in df.columns:
                continue

            series = df[macro]

            # heuristic: treat big macro levels differently
            if macro in ["GDP", "Industrial_Production", "Retail_Sales", "PCE"]:
                # additive shock
                df[macro] = series + shock

            elif macro in ["CPI", "Core_CPI", "PPI", "Oil_WTI", "Copper"]:
                # multiplicative shock (percentage move)
                df[macro] = series * (1 + shock)

            elif macro in ["Fed_Funds_Rate", "10Y_Treasury", "2Y_Treasury", "Yield_Spread"]:
                # rate shocks -> additive
                df[macro] = series + shock

            else:
                # default safe additive
                df[macro] = series + shock

        return df
    
    def compare(self):
        if self.baseline_predictions is None:
            self.run_baseline()

        comparison = pd.DataFrame({
            "baseline": self.baseline_predictions,
            "scenario": self.scenario_predictions
        })

        comparison["impact"] = comparison["scenario"] - comparison["baseline"]

        return comparison

def create_dataset(ticker):
    # create master dataframe
    feature_engineer = FeatureEngineer(ticker)
    master_df = feature_engineer.load_data()
    lagged_df = feature_engineer.apply_lags(master_df)
    target_df = feature_engineer.create_target(lagged_df)

    return target_df

def create_rf_model(target_df):
    # create random forest model
    rf_model = RandomForestModel()
    rf = rf_model.run_random_forest(target_df)
    metrics = rf_model.get_metrics()
    gini_feat = rf_model.feature_importance_gini()
    perm_feat = rf_model.feature_importance_permutation()

    return rf, metrics

def create_scenarios(rf_model, target_df, scenario):
    # run scenario engine on random forest model
    scenario_engine = ScenarioEngine(rf_model, target_df)
    baseline = scenario_engine.run_baseline()
    predef_scenario = scenario_engine.run_predefined_scenario(scenario)
    compare = scenario_engine.compare()

    return compare

ticker = 'XLK'
scenario = 'inflation_spike'
df = create_dataset(ticker)
rf_model, metrics = create_rf_model(df)
scenarios = create_scenarios(rf_model, df, scenario)

print(metrics)
print(scenarios)

def plot_individual_tree(regressor: RandomForestRegressor, features: list):
    """ Visualizes an individual tree in the forest, showing the decision making at each step """
   
    tree_to_plot = regressor.estimators_[0]
    plt.figure(figsize=(20,10))
    plot_tree(tree_to_plot, feature_names=features, filled=True, rounded=True, fontsize=10)
    plt.title("Decision Tree from Random Forest")
    plt.show()