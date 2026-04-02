import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.stats.anova import anova_lm
import os


class LinearRegressionModel:
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

    def linear_regression(self):
        '''
        Normal OLS regression using statsmodel. Returns model summary and ANOVA table.
        '''
        predictors = " + ".join(self.x.columns)
        formula = f"y ~ {predictors}"

        model = smf.ols(formula, data=self.train).fit()
        y_pred = model.predict(self.test)

        results, directional_accuracy, r2_oos = self._model_testing(model, self.test)
        self._graph(results, self.train, self.test, y_pred, self.etf, self.output_dir, directional_accuracy, r2_oos)

        print(model.summary())
        print(anova_lm(model, typ=1))

        return model.summary(), anova_lm(model, typ=1)

    def recursive_ordinary_least_squares(self):
        '''
        Implements Recursive OLS regression. Returns model summary and predictions.
        '''
        predictors = " + ".join(self.x.columns)
        formula = f"y ~ {predictors}"

        reg_rls = sm.RecursiveLS.from_formula(formula, data=self.train)
        model_rls = reg_rls.fit()

        coeffs_over_time = pd.DataFrame(
            model_rls.filtered_state.T,
            columns=["Intercept"] + list(self.x.columns),
            index=self.train.index,
        )

        y_pred = []
        coefs = model_rls.params.copy()

        X_test = sm.add_constant(self.test[self.x.columns])

        for t in range(len(self.test)):
            x_t = X_test.iloc[t].values
            y_t_pred = x_t @ coefs
            y_pred.append(y_t_pred)

            tmp_df = pd.concat([self.train, self.test.iloc[: t + 1]], axis=0)
            reg_tmp = sm.RecursiveLS.from_formula(formula, data=tmp_df)
            model_tmp = reg_tmp.fit()
            coefs = model_tmp.params

        y_pred = pd.Series(y_pred, index=self.test.index)

        fig, axes = plt.subplots(
            len(self.x.columns) + 1,
            1,
            figsize=(10, 3 * (len(self.x.columns) + 1)),
            sharex=True,
        )

        for i, col in enumerate(coeffs_over_time.columns):
            axes[i].plot(coeffs_over_time.index, coeffs_over_time[col], label=f"{col} (train)")
            axes[i].plot(self.test.index, [coefs[i]] * len(self.test), "--", label=f"{col} (test start)")
            axes[i].set_ylabel(col)
            axes[i].legend(loc="upper right")

        plt.suptitle(f"Recursive OLS - {self.etf}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()
        plt.savefig(os.path.join(self.output_dir, f"RLS_{self.etf}.png"))
        plt.close(fig)

        return model_rls.summary(), y_pred

    def window_ordinary_least_squares(self, window=120):
        '''
        Implements Windowed OLS regression. Returns predictions and performance metrics.
        '''
        predictors = " + ".join(self.x.columns)
        formula = f"y ~ {predictors}"

        full_data = pd.concat([self.train, self.test])

        reg_window = RollingOLS.from_formula(formula, window=window, data=full_data)
        model_window = reg_window.fit()

        y_pred = []
        for idx in self.test.index:
            coefs = model_window.params.loc[:idx].iloc[-1].values
            x_t = np.r_[1, self.test.loc[idx, self.x.columns].values]
            y_t_pred = x_t @ coefs
            y_pred.append(y_t_pred)

        results_df, directional_accuracy, r2_oos = self._rls_model_testing(self.test, y_pred)
        self._rls_graph(results_df, self.train, self.test, y_pred, self.etf, self.output_dir, directional_accuracy, r2_oos)

        return results_df, directional_accuracy, r2_oos

    def _rls_graph(self, results_df, train, test, y_pred, etf_name="ETF", output_dir="reports/images", directional_accuracy=None, r2_oos=None):
        '''
        Graphs the results of RLS or Windowed OLS testing.
        '''
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(14, 6))

        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(train.index, train["y"], label="Train (Actual)", color="black")
        ax1.plot(test.index, test["y"], label="Test (Actual)", linewidth=2, color="orange")
        ax1.plot(test.index, y_pred, label="Test (Predicted)", linestyle="--", color="green")
        ax1.axvline(test.index[0], color="black", linestyle=":", label="Train/Test Split")
        ax1.set_title("Train/Test with Predictions")
        ax1.set_xlabel("Month")
        ax1.set_ylabel("Value")
        ax1.legend()
        ax1.grid(True)

        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(results_df.index, results_df["Actual"], label="Actual", linewidth=2, color="orange")
        ax2.plot(results_df.index, results_df["Predicted"], label="Predicted", linestyle="--", color="green")
        ax2.set_title("Out-of-Sample: Actual vs Predicted")
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Value")
        ax2.legend()
        ax2.grid(True)

        if directional_accuracy is not None and r2_oos is not None:
            metrics_text = f"R² (OOS): {r2_oos:.4f}\nDirectional Accuracy: {directional_accuracy:.2%}"
            ax2.text(
                0.02,
                0.98,
                metrics_text,
                transform=ax2.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        plt.tight_layout()
        plt.show()
        plt.close()

    def _rls_model_testing(self, test, y_pred):
        '''
        Evaluates RLS or Windowed OLS predictions. Returns results dataframe, directional accuracy, and out-of-sample R².
        '''
        results_df = test.copy()
        results_df = results_df.rename(columns={"y": "Actual"})
        results_df["Predicted"] = y_pred

        results_df["Error"] = results_df["Actual"] - results_df["Predicted"]
        results_df["Squared_Error"] = results_df["Error"] ** 2

        results_df["Actual_Change"] = results_df["Actual"].diff()
        results_df["Predicted_Change"] = results_df["Predicted"].diff()
        results_df = results_df.iloc[1:]

        results_df["Actual_Direction"] = results_df["Actual_Change"] > 0
        results_df["Predicted_Direction"] = results_df["Predicted_Change"] > 0
        results_df["Correct_Direction"] = results_df["Actual_Direction"] == results_df["Predicted_Direction"]
        results_df["Direction_Label"] = results_df["Correct_Direction"].map({True: "Yes", False: "No"})

        directional_accuracy = results_df["Correct_Direction"].mean()

        mean_train = results_df["Actual"].mean()
        sse_model = results_df["Squared_Error"].sum()
        sse_mean = ((results_df["Actual"] - mean_train) ** 2).sum()
        r2_oos = 1 - (sse_model / sse_mean)

        results_df.drop(
            ["Correct_Direction", "Predicted_Direction", "Squared_Error", "Actual_Change", "Actual_Direction"],
            axis=1,
            inplace=True,
        )

        return results_df, directional_accuracy, r2_oos

    def _model_testing(self, model, test):
        '''
        Evaluates OLS predictions. Returns results dataframe, directional accuracy, and out-of-sample R².
        '''
        test = test.copy()
        results = []
        for idx, row in test.iterrows():
            y_actual = row["y"]
            y_pred = model.predict(row.to_frame().T).iloc[0]
            results.append({"Month": idx, "Actual": y_actual, "Predicted": y_pred})

        results_df = pd.DataFrame(results)
        results_df.set_index("Month", inplace=True)

        results_df["Error"] = results_df["Actual"] - results_df["Predicted"]
        results_df["Squared_Error"] = results_df["Error"] ** 2

        results_df["Actual_Change"] = results_df["Actual"].diff()
        results_df["Predicted_Change"] = results_df["Predicted"].diff()
        results_df = results_df.iloc[1:]

        results_df["Actual_Direction"] = results_df["Actual_Change"] > 0
        results_df["Predicted_Direction"] = results_df["Predicted_Change"] > 0
        results_df["Correct_Direction"] = results_df["Actual_Direction"] == results_df["Predicted_Direction"]
        results_df["Direction_Label"] = results_df["Correct_Direction"].map({True: "Yes", False: "No"})

        directional_accuracy = results_df["Correct_Direction"].mean()

        pos = results_df[results_df["Actual_Change"] > 0]["Actual_Change"]
        neg = results_df[results_df["Actual_Change"] < 0]["Actual_Change"]

        pos_low, pos_high = pos.quantile([0.33, 0.66])
        neg_low, neg_high = neg.quantile([0.33, 0.66])

        results_df["Actual_LMH_Dir"] = results_df["Actual_Change"].apply(
            lambda x_val: self._directional_lmh(x_val, pos_low, pos_high, neg_low, neg_high)
        )
        results_df["Predicted_LMH_Dir"] = results_df["Predicted_Change"].apply(
            lambda x_val: self._directional_lmh(x_val, pos_low, pos_high, neg_low, neg_high)
        )

        labels = ["Bull_Low", "Bull_Medium", "Bull_High", "Bear_Low", "Bear_Medium", "Bear_High"]
        cm = confusion_matrix(results_df["Actual_LMH_Dir"], results_df["Predicted_LMH_Dir"], labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)

        directional_per_class = {}
        for label in labels:
            subset = results_df[results_df["Actual_LMH_Dir"] == label]
            directional_per_class[label] = subset["Actual_LMH_Dir"].eq(subset["Predicted_LMH_Dir"]).mean() if len(subset) > 0 else np.nan

        actual_counts = results_df["Actual_LMH_Dir"].value_counts().reindex(labels, fill_value=0)
        pred_counts = results_df["Predicted_LMH_Dir"].value_counts().reindex(labels, fill_value=0)
        counts_df = pd.DataFrame({"Actual_Count": actual_counts, "Predicted_Count": pred_counts})

        print("Counts per Class:")
        print(counts_df)

        mean_train = model.model.endog.mean()
        sse_model = results_df["Squared_Error"].sum()
        sse_mean = ((results_df["Actual"] - mean_train) ** 2).sum()
        r2_oos = 1 - (sse_model / sse_mean)

        results_df.drop(
            ["Correct_Direction", "Predicted_Direction", "Squared_Error", "Actual_Change", "Actual_Direction"],
            axis=1,
            inplace=True,
        )

        print(f"Out-of-sample R²: {r2_oos:.4f}")
        print(f"Directional Accuracy: {directional_accuracy:.2%}")

        return results_df, directional_accuracy, r2_oos

    @staticmethod
    def _directional_lmh(change, pos_low, pos_high, neg_low, neg_high):
        '''
        Classifies change into directional LMH buckets.
        '''
        if change > 0:
            if change <= pos_low:
                return "Bull_Low"
            elif change <= pos_high:
                return "Bull_Medium"
            else:
                return "Bull_High"
        elif change < 0:
            if change >= neg_high:
                return "Bear_Low"
            elif change >= neg_low:
                return "Bear_Medium"
            else:
                return "Bear_High"
        else:
            return "Neutral"

    def _graph(self, df, train, test, y_pred, etf, output_dir, directional_accuracy, r2_oos):
        '''
        Graphs the results of OLS testing.
        '''
        plt.figure(figsize=(14, 6))

        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(train.index, train["y"], label="Train (Actual)", color="black")
        ax1.plot(test.index, test["y"], label="Test (Actual)", linewidth=2, color="orange")
        ax1.plot(test.index, y_pred, label="Test (Predicted)", linestyle="--", color="green")
        ax1.axvline(test.index[0], color="black", linestyle=":", label="Train/Test Split")
        ax1.set_title("Train/Test with Predictions")
        ax1.set_xlabel("Month")
        ax1.set_ylabel("Value")
        ax1.legend()
        ax1.grid(True)

        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(df.index, df["Actual"], label="Actual", linewidth=2, color="orange")
        ax2.plot(df.index, df["Predicted"], label="Predicted", linestyle="--", color="green")
        ax2.set_title("Out-of-Sample: Actual vs Predicted")
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Value")
        ax2.legend()
        ax2.grid(True)

        metrics_text = f"R² (OOS): {r2_oos:.4f}\nDirectional Accuracy: {directional_accuracy:.2%}"
        ax2.text(
            0.02,
            0.98,
            metrics_text,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        etf_name = os.path.basename(etf).replace(".csv", "")
        filepath = os.path.join(output_dir, f"{etf_name}_results.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()


# Legacy wrappers for backward compatibility in case something breaks. Will need to tell mark to update app.py

def linear_regression(x, y, etf, output_dir="reports/images"):
    return LinearRegressionModel(x, y, etf, output_dir).linear_regression()


def recursive_ordinary_least_squares(x, y, etf, output_dir="reports/images"):
    return LinearRegressionModel(x, y, etf, output_dir).recursive_ordinary_least_squares()


def window_ordinary_least_squares(x, y, etf, output_dir="reports/images", window=120):
    return LinearRegressionModel(x, y, etf, output_dir).window_ordinary_least_squares(window=window)