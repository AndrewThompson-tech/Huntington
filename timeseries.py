import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
import os

from data_cleanse import *



from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA


'''
Do not use ARIMA to predict ETFs themsevles, this is useless since datas have a random walk (Efficient market hypothesis)
Instead, lets predict the macro vaiables since they do have seasonality and economies have natural cycles. 
Then we can keep this in mind when making decision on what datas to buy.

This is useful beacause while our linear regression is decent, what would really give an edge is identifying macro
surprises.
    For example, if we predict unemploment rates to be much higher than consensus, then we can expect a market downturn and adjust our portfolio accordingly.
    Relating this into our linear regession model, XLV, XLY, XLP (health care, consumer discretionary, consumer staples) use UNRATE. We can
    make decisions on how much to allocate to these sectors based on our predictions for UNRATE. Not only should we implemnt the ETF-macro lag, but we should
    also look beyond that.

    *Note: Not all macros are naturally seasonal, like policy driven Fed rates.
    **Note: Some macros are already seasonally adjusted. 
'''


# # https://www.geeksforgeeks.org/machine-learning/time-series-analysis-and-forecasting/


# # data = 'data/raw_data/datas/XLP_monthly.csv'
# macro = 'data/raw_data/MCOILWTICO.csv'
# data = fix_pd(macro)
# # data = interpolate_monthly(data)
# data = log_diff(data) 
# data = data[:240]



# y = data.dropna()

# # Returns an object with seasonal, trend, and resid attributes.
# decomp = seasonal_decompose(y, model="additive", period=12)
# decomp.plot()
# plt.show()


# train = y[:int(len(y)*0.8)]
# test = y[int(len(y)*0.8):]
# # test = test.shift(1).dropna()


# model = pm.auto_arima(
#     train,
#     seasonal=True,
#     m=12,
#     trace=True,
#     # error_action="ignore",
#     # suppress_warnings=True
# )

# print(model.summary())

# forecast = model.predict(n_periods=len(test))
# forecast = pd.Series(forecast, index=test.index)

# plt.figure(figsize=(12,5))
# plt.plot(train, label='Training Data')
# plt.plot(test, label='Actual Values', linewidth=2)
# plt.plot(forecast, label='Forecasted Values', linestyle="--")
# plt.title("Forecast vs Actual")
# plt.xlabel("Time")
# plt.ylabel("Value")
# plt.legend()
# plt.grid(True)
# plt.show()

# mae = mean_absolute_error(test, forecast)
# rmse = np.sqrt(mean_squared_error(test, forecast))

# naive = test.shift(1).fillna(0)

# print("Naive MAE:", mean_absolute_error(test, naive))
# print("Naive RMSE:", np.sqrt(mean_squared_error(test, naive)))



class ARMAFamily:
    def __init__(self, x, y, etf, output_dir="reports/images", train_ratio=0.80):
        self.x = x.copy()
        self.y = y
        self.etf = etf
        self.output_dir = output_dir
        self.train_ratio = train_ratio

        os.makedirs(self.output_dir, exist_ok=True)

        self.df = self.x.copy()
        self.df["y"] = self.y

        self.train_size = int(len(self.df) * self.train_ratio)
        self.train = self.df.iloc[: self.train_size]
        self.test = self.df.iloc[self.train_size :]

            
    def AR(self, lags=6):
        '''
        Autorregressive model, where the prediction is based on the past values of the series.
        For examample, we can predict next month's GDP based on the past 6 months of GDP data. 
        This is useful for macros that have strong autocorrelation.
        '''
        model = AutoReg(self.train["y"], lags = lags).fit()
        pred = model.predict(start=self.test.index[0], end=self.test.index[-1])
        return pred, model


    def MA(self, order=1):
        '''
        Moving Average model, where the prediction is based on the past forecast errors. 
        For example, we can predict next month's GDP based on the past 6 months of forecast errors. 
        This is useful for macros that have strong autocorrelation in the errors.
        '''        
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(self.train["y"], order=(0,0,order)).fit()
        pred = model.predict(start=self.test.index[0], end=self.test.index[-1])
        return pred, model

    def ARIMA(self, order=(1,1,1)):
        '''
        Combines AR and MA, and also includes differencing to make the series stationary. 
        This is useful for macros that have both autocorrelation and non-stationarity. 
        '''
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(self.train["y"], order=order).fit()
        pred = model.predict(start=self.test.index[0], end=self.test.index[-1], typ='levels')
        return pred, model

    def ARIMAX(self, order=(2,1,2), exog_lags=6):
        '''
        ARIMA with exogenous variables, where we can include other macro variables as predictors. 
        For example, we can predict next month's GDP based on the past 6 months of GDP data and the past 6 months of unemployment rate data. 
        This is useful for macros that are influenced by other macros. 
        '''
        exog_train = self.train[self.x.columns].iloc[-len(self.train):]
        exog_test = self.test[self.x.columns].iloc[:len(self.test)]
        model = ARIMA(self.train["y"], order=order, exog=exog_train).fit()
        pred = model.predict(start=self.test.index[0], end=self.test.index[-1], exog=exog_test)
        return pred, model
    
    def AUTO_ARIMA(self, seasonal=True, m=12):
        '''
        Automatically selects the best ARIMA model based on AIC/BIC. 
        This is useful for quickly identifying the best model without having to manually test different combinations of p, d, q parameters. 
        '''
        model = pm.auto_arima(
            self.train["y"],
            seasonal=seasonal,
            m=m,
            trace=True,
            error_action="ignore",
            suppress_warnings=True,
        )

        # model = pm.auto_arima(
        #     y=self.train["y"],        # your macro series
        #     X=self.train.get("X", None), # optional exogenous variables
        #     start_p=1, start_q=1,     # lower start to reduce computation
        #     max_p=3, max_q=3,         # macros rarely need high AR/MA orders
        #     d=None,                    # let auto_arima determine differencing
        #     seasonal=True,
        #     m=12,                      # monthly data -> yearly seasonality
        #     start_P=0, max_P=1,        # seasonal AR, low to prevent overfitting
        #     D=None,                     # let seasonal differencing be estimated
        #     max_D=1,
        #     stepwise=True,             # faster search
        #     information_criterion='aic', # common choice
        #     suppress_warnings=True,
        #     error_action='ignore',
        #     trace=True,
        #     with_intercept="auto",
        #     max_order=6                # keep total p+q+P+Q reasonable
        # )
        pred = pd.Series(model.predict(n_periods=len(self.test)), index=self.test.index)
        return pred, model



    def NARX(self, lags=6, nonlinear_func=np.tanh):
        '''
        Nonlinear Autoregressive Exogenous model, where we can include nonlinear relationships between the variables. 
        For example, we can predict next month's GDP based on the past 6 months of GDP data and the past 6 months of unemployment rate data, but also include a nonlinear term for the interaction between GDP and unemployment rate. 
        This is useful for macros that have complex relationships with other macros. 
        https://en.wikipedia.org/wiki/Nonlinear_autoregressive_exogenous_model
        '''
                # Simple NARX implementation using past y and exogenous variables
        X = []
        Y = []
        y_series = self.df["y"].values
        exog = self.df[self.x.columns].values if not self.x.empty else np.zeros((len(y_series),0))
        for i in range(lags, len(y_series)):
            row = y_series[i-lags:i].tolist() + exog[i-lags:i].flatten().tolist()
            X.append(row)
            Y.append(y_series[i])
        X = np.array(X)
        Y = np.array(Y)
        # Nonlinear regression (ridge) on transformed inputs
        from sklearn.linear_model import Ridge
        X_nl = nonlinear_func(X)
        model = Ridge().fit(X_nl[:self.train_size-lags], Y[:self.train_size-lags])
        X_test_nl = nonlinear_func(X[self.train_size-lags:])
        pred = model.predict(X_test_nl)
        pred = pd.Series(pred, index=self.test.index)
        return pred, model
   
    @staticmethod
    def generate_macro_surprises(MACRO_df):
        """
            Takes raw macro input
            
            For each macro column:
            - Fit ARIMA
            - Generate in-sample forecast (rolling would be better later)
            - Compute surprise = actual - expected
            
            Returns:
            - DataFrame of macro surprises
        """
        surprise_df = pd.DataFrame(index=MACRO_df.index)

        for col in MACRO_df.columns:
            series = MACRO_df[col].dropna()

            if len(series) < 50:
                continue  # skip short series

            try:
                arma = ARMAFamily(x=pd.DataFrame(), y=series, etf=col)
                forecast, fitted_model = arma.AUTO_ARIMA()  # fitted_model is your trained model

                # Use in-sample predictions for all indices
                in_sample_pred = pd.Series(fitted_model.predict_in_sample(), index=series.index)
                surprise = series - in_sample_pred
                surprise_df[col + "_surprise"] = surprise

            except Exception as e:
                print(f"ARIMA failed for {col}: {e}")

        print(surprise_df.head())       

        return surprise_df

    def plot_forecast(self, forecast, model_name="Model", save=True):
        """
        Plots training, test, and forecasted values in a two-panel format.
        
        Parameters:
        - forecast: pd.Series of forecasted values (aligned with self.test index)
        - model_name: str, name of the model (used for title and saving)
        - save: bool, whether to save the figure to self.output_dir
        """
        # Compute metrics
        actual = self.test["y"]


        plt.figure(figsize=(14, 6))

        # --- Panel 1: Train/Test with Forecast ---
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(self.train.index, self.train["y"], label="Train (Actual)", color="black")
        ax1.plot(self.test.index, actual, label="Test (Actual)", linewidth=2, color="orange")
        ax1.plot(self.test.index, forecast, label="Test (Forecasted)", linestyle="--", color="green")
        ax1.axvline(self.test.index[0], color="black", linestyle=":", label="Train/Test Split")
        ax1.set_title(f"Train/Test with Forecast - {model_name}")
        ax1.set_xlabel("Month")
        ax1.set_ylabel("Value")
        ax1.legend()
        ax1.grid(True)

        # --- Panel 2: Out-of-Sample Actual vs Forecasted ---
        df_oos = pd.DataFrame({"Actual": actual.values, "Predicted": forecast.values}, index=self.test.index)

        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(df_oos.index, df_oos["Actual"], label="Actual", linewidth=2, color="orange")
        ax2.plot(df_oos.index, df_oos["Predicted"], label="Forecasted", linestyle="--", color="green")
        ax2.set_title("Out-of-Sample: Actual vs Forecasted")
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Value")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if save:
            os.makedirs(self.output_dir, exist_ok=True)
            filepath = os.path.join(self.output_dir, f"{self.etf}_{model_name}_forecast.png")
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Saved plot to {filepath}")

        plt.show()
        plt.close()


    def evaluate_forecast(self, forecast, actual=None):
        """
        Evaluates a forecast series against actuals.
        
        Parameters:
        - forecast: pd.Series of predicted values (aligned with self.test.index)
        - actual: optional pd.Series; defaults to self.test["y"]
        
        Returns:
        - results_df: DataFrame with actual, predicted, error, LMH direction
        - directional_accuracy: float (0-1)
        - r2_oos: out-of-sample R²
        """
        if actual is None:
            actual = self.test["y"]
        
        results_df = pd.DataFrame({
            "Actual": actual,
            "Predicted": forecast
        })
        
        results_df["Error"] = results_df["Actual"] - results_df["Predicted"]
        results_df["Squared_Error"] = results_df["Error"] ** 2
        
        # Compute directional changes
        results_df["Actual_Change"] = results_df["Actual"].diff()
        results_df["Predicted_Change"] = results_df["Predicted"].diff()
        results_df = results_df.iloc[1:]  # remove first NaN
        
        results_df["Actual_Direction"] = results_df["Actual_Change"] > 0
        results_df["Predicted_Direction"] = results_df["Predicted_Change"] > 0
        results_df["Correct_Direction"] = results_df["Actual_Direction"] == results_df["Predicted_Direction"]
        directional_accuracy = results_df["Correct_Direction"].mean()

    
        # Out-of-sample R²
        mean_train = self.train["y"].mean()
        sse_model = results_df["Squared_Error"].sum()
        sse_mean = ((results_df["Actual"] - mean_train) ** 2).sum()
        r2_oos = 1 - (sse_model / sse_mean)
        
        print(f"Out-of-sample R²: {r2_oos:.4f}")
        print(f"Directional Accuracy: {directional_accuracy:.2%}")
        
        # Keep only useful columns
        results_df.drop(
            ["Correct_Direction", "Predicted_Direction", "Squared_Error", "Actual_Change", "Actual_Direction"],
            axis=1,
            inplace=True,
        )
        
        return results_df, directional_accuracy, r2_oos


if __name__ == "__main__":
    # Example usage
    # macro = 'data/raw_data/INDPRO.csv'
    etf = 'data/raw_data/ETFs/XLV_monthly.csv'

    data = fix_pd(etf)
    data = data["Close"].dropna()
    # data = log_diff(data)
    data = data[:240]

    # n = 240
    # t = np.arange(n)

    # # Strong AR structure, gentle decay, minimal noise
    # data = (np.exp(-0.005 * t)) * np.sin( * np.pi * t / 12) + np.random.normal(0, 0.05, n)

    # decomp = seasonal_decompose(data, model="additive", period=12)
    # decomp.plot()
    # plt.show()


    model = ARMAFamily(x=pd.DataFrame(), y=data, etf="XLE")
    forecast, fitted_model = model.AUTO_ARIMA()
    model.plot_forecast(forecast, model_name="AUTO_ARIMA")
    print(fitted_model.summary())
    model.evaluate_forecast(forecast)
