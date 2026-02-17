import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


def linear_regression(x, y):

    # Combine into one DataFrame for formula API
    df = x.copy()
    df["y"] = y

    # Train/test split
    train_size = int(len(df) * 0.8)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]

    # Build formula string automatically
    predictors = " + ".join(x.columns)
    formula = f"y ~ {predictors}"

    # Fit OLS model
    model = smf.ols(formula, data=train).fit()

    # OLS summary and ANOVA table
    print(model.summary())
    print(anova_lm(model, typ=1)) 
    # It would be nice to pull individual pieces of these and perform a check/verification for model health

    # Predictions
    y_pred = model.predict(test)

    # Metrics
    mse = mean_squared_error(test["y"], y_pred)
    r2 = r2_score(test["y"], y_pred)

    print("Mean Squared Error:", mse)
    print("R^2 Score:", r2)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train.index, train["y"], label="Train (Actual)", alpha=0.6)
    plt.plot(test.index, test["y"], label="Test (Actual)", linewidth=2)
    plt.plot(test.index, y_pred, label="Test (Predicted)", linestyle="--")

    plt.axvline(test.index[0], color="black", linestyle=":", label="Train/Test Split")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
