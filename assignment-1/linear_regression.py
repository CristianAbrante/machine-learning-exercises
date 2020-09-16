import data_reader as dr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def get_regression_model():
    x, y = dr.read_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    reg = LinearRegression(fit_intercept=False).fit(x_train, y_train)

    return reg, x_train, x_test, y_train, y_test
