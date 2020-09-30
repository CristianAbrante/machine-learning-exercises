import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import math


def vc_bound_score(empirical_error, d, m, delta):
    return empirical_error \
           + np.sqrt((2 * np.log(math.e * m / d)) / (m / d)) \
           + np.sqrt(np.log(1 / delta) / (m / d))


if __name__ == '__main__':
    np.random.seed(42)

    n_tots = range(20, 201)

    training_errors = np.array([])
    test_errors = np.array([])
    vc_bounds = np.array([])

    for n_tot in n_tots:
        n = int(n_tot / 2)  # will use half in training, half in testing

        # two blobs; labels are 0 and 1
        X, y = make_blobs(n_tot, centers=2, cluster_std=4.0, random_state=1)

        # divide into training and testing
        order = np.random.permutation(n_tot)
        train = order[:n]  # these will be the training samples
        test = order[n:]  # and these are for testing

        x_train = X[train, :]
        y_train = y[train]
        x_test = X[test, :]
        y_test = y[test]

        my_classifier = Perceptron()

        # this is how to train the model with data in training set
        my_classifier.fit(x_train, y_train)

        # calculate predictions
        y_pred_train = my_classifier.predict(x_train)
        y_pred_test = my_classifier.predict(x_test)

        # Calculate error for train and test
        error_train = 1 - accuracy_score(y_train, y_pred_train, normalize=False) / float(y_train.size)
        error_test = 1 - accuracy_score(y_test, y_pred_test, normalize=False) / float(y_test.size)

        # Calculate VC Bound
        d = X.shape[1] + 1  # number of features + 1 for perceptron
        m = len(y_test)  # number of samples
        delta = 0.08
        vc_bound = vc_bound_score(empirical_error=error_test, d=d, m=m, delta=delta)

        # Append to total errors
        training_errors = np.append(training_errors, error_train)
        test_errors = np.append(test_errors, error_test)
        vc_bounds = np.append(vc_bounds, vc_bound)

        print(
            f"N tot -> {n_tot} | training error -> {np.round(error_train, 2)} | test error -> {np.round(error_test, 2)} | VC bound -> {np.round(vc_bound, 4)}")

    plt.plot(n_tots, training_errors, 'r', label="training error")
    plt.plot(n_tots, test_errors, 'b', label="test error")
    plt.plot(n_tots, vc_bounds, 'g', label="VC Bound")
    plt.xlabel("Number of data points")
    plt.ylabel("Error")
    plt.legend(loc='upper right', frameon=False)
    plt.show()
