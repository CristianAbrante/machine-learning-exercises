import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


DATA_FOLDER = "assignment-1/data/"
INPUT_FILE = "X.csv"
TARGET_FILE = "Y.csv"


def read_data(data_folder, input_file, target_file):
    data_path = Path(data_folder)
    input = pd.read_csv(data_path / input_file)
    target = pd.read_csv(data_path / target_file)
    return input, target


if __name__ == '__main__':
    X, Y = read_data(DATA_FOLDER, INPUT_FILE, TARGET_FILE)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

    reg = LinearRegression(fit_intercept=False).fit(X_train, Y_train)

    Y_pred = reg.predict(X_test)

    print(np.sqrt(mean_squared_error(Y_test, Y_pred)))