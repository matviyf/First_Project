from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from numpy.typing import ArrayLike
from joblib import dump


def train_regression_model(X_train: ArrayLike, y_train: ArrayLike) -> LinearRegression:
    """
    Train a regression model using the provided training data.

    This function takes training data consisting of a feature matrix 'X_train' and
    corresponding target values 'y_train', and trains a linear regression model.
    The trained model is returned for further use.

    Args:
        X_train (array-like): Training feature matrix.
        y_train (array-like): Target values for training.

    Returns:
        sklearn.linear_model.LinearRegression: Trained linear regression model.

    """

    # TODO: your code here
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

def save_regression_model(model: LinearRegression, filename: str = "linear_regression_model.joblib"):
    """
    Serialize and save the regression model.

    This function takes a trained regression 'model' and file name 'filename' that has a default value.
    It uses Joblib 'dump' to save the model using the provided file name.

    Args:
        model (sklearn.linear_model.LinearRegression): Trained regression model to be evaluated.
        filename (str): Name of the file that is used to store the model.

    """
    
    # TODO: your code here
    dump(model, filename)

def evaluate_regression_model(model: LinearRegression, X_test: ArrayLike, y_test: ArrayLike):
    """
    Evaluate the performance of a regression model on test data.

    This function takes a trained regression 'model', test feature matrix 'X_test',
    and corresponding test target values 'y_test'. It calculates Mean Squared Error (MSE)
    and prints it in terminal.

    Args:
        model (sklearn.linear_model.LinearRegression): Trained regression model to be evaluated.
        X_test (array-like): Test feature matrix.
        y_test (array-like): Validation target values.

    """
    
    # TODO: your code here
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")

def save_initial_datasets(X: ArrayLike, y: ArrayLike):
    """
    Serialize and save datasets.

    This function takes entire feature matrix 'X', and corresponding target values 'y'.
    It uses Joblib 'dump' to save both arrays as predefined files.

    Args:
        X (array-like): Test feature matrix.
        y (array-like): Validation target values.

    """
    X_filename = "X.joblib"
    y_filename = "y.joblib"
    
    # TODO: your code here
    dump(X, X_filename)
    dump(y, y_filename)
if __name__ == '__main__':
    if __name__ == '__main__':

    # Load real apartment rental data
        file_path = Path(__file__).parent / "apartments_10K.csv"

    data = pd.read_csv(
        file_path,
        sep=";",
        encoding="cp1252"
    )

    # Filter monthly rental listings
    clean_data = data[data["price_type"] == "Monthly"].copy()

    # Remove extreme values
    clean_data = clean_data[
        (clean_data["square_feet"] >= 200) &
        (clean_data["square_feet"] <= 3000) &
        (clean_data["price"] >= 300) &
        (clean_data["price"] <= 10000)
    ]

    # Prepare features and target
    X = clean_data[["square_feet"]].to_numpy()
    y = clean_data["price"].to_numpy()

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a linear regression model
    model = train_regression_model(X_train, y_train)

    # Evaluate the model
    evaluate_regression_model(model, X_test, y_test)

    # Save the model
    save_regression_model(model)

    # Save datasets
    save_initial_datasets(X, y)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = train_regression_model(X_train, y_train)

    # Evaluate the model
    evaluate_regression_model(model, X_test, y_test)

    # Save the model
    save_regression_model(model)

    # Save datasets
    save_initial_datasets(X, y)