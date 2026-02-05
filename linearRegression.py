import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def linear_regression(x, y):
    '''
    Simple liear regression,
         https://www.geeksforgeeks.org/machine-learning/multiple-linear-regression-with-scikit-learn/
    
    :param x: macro variabes in a dataframe
    :param y: ETF values
    '''

    # Train / test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, shuffle=False
    )

    # Train linear regression model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Predictions
    y_pred = model.predict(x_test)


    # mean squared error and r2
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R^2 Score:", r2)

    # Standardized coefficients
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model_scaled = LinearRegression()
    model_scaled.fit(x_scaled, y)
    std_coeffs = pd.Series(model_scaled.coef_, index=x.columns)
    print(f"\nStandardized Coefficients:\n{std_coeffs}\n")

    # Coefficients
    coefficients = pd.Series(model.coef_, index=x.columns)
    print(f"Model Coefficients:\n{coefficients}")

    # Graph
    plt.figure(figsize=(10, 6))
    plt.plot(y_train.index, y_train, label="Train (Actual)", alpha=0.6)
    plt.plot(y_test.index, y_test, label="Test (Actual)", linewidth=2)
    plt.plot(y_test.index, y_pred, label="Test (Predicted)", linestyle="--")

    plt.axvline(y_test.index[0], color="black", linestyle=":", label="Train/Test Split")

    plt.title("Closing Price: Train vs Test")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()