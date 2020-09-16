import numpy as np
import linear_regression as lr
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    reg, x_train, x_test, y_train, y_test = lr.get_regression_model()
    y_pred = reg.predict(x_test)

    print("Exercise 4:")
    print(np.sqrt(mean_squared_error(y_test, y_pred)))
