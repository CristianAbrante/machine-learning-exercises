import numpy as np
import data_reader as dr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

if __name__ == '__main__':
    x, y = dr.read_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    print("Exercise 4:")
    reg = LinearRegression(fit_intercept=False).fit(x_train, y_train)
    y_predicted = reg.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
    print 'RMSR = ', np.round(rmse, 2)

    print("-----------")

    print("Exercise 5:")
    # TODO: Do it with k=5 also
    kf_x = x_train
    kf_y = y_train

    k_values = [2, 5]
    for k in k_values:
        print "K = ", k
        kf = KFold(n_splits=k, random_state=1, shuffle=True)
        kf_rmse = np.array([])

        for train_index, test_index in kf.split(kf_x):
            kf_x_train, kf_x_test = kf_x.iloc[train_index], kf_x.iloc[test_index]
            kf_y_train, kf_y_test = kf_y.iloc[train_index], kf_y.iloc[test_index]

            kf_reg = LinearRegression(fit_intercept=False).fit(kf_x_train, kf_y_train)
            kf_y_predicted = kf_reg.predict(kf_x_test)

            rmse = np.sqrt(mean_squared_error(kf_y_test, kf_y_predicted))
            kf_rmse = np.append(kf_rmse, rmse)

        # avg_rmse = np.sqrt(np.sum(np.power(kf_rmse, 2) / k))
        avg_rmse = np.mean(kf_rmse)
        var_rmse = np.var(kf_rmse)
        print "avg rmse: ", np.round(avg_rmse, 2), "|  var rmse: ", np.round(var_rmse, 2)
