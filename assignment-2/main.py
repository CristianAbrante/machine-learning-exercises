import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    n_tot = 50  # choose the number of samples to be generated

    n = int(n_tot / 2)  # will use half in training, half in testing

    # two blobs; labels are 0 and 1
    X, y = make_blobs(n_tot, centers=2, cluster_std=4.0, random_state=1)

    # divide into training and testing
    np.random.seed(42)
    order = np.random.permutation(n_tot)
    train = order[:n]  # these will be the training samples
    test = order[n:]  # and these are for testing

    my_classifier = Perceptron()
    # this is how to train the model with data in training set
    my_classifier.fit(X[train, :], y[train])
    # and this is how to predict the labels for test data
    predictions = my_classifier.predict(X[test, :])

    training_error = 1 - accuracy_score(y[test], predictions, normalize=False) / float(y[train].size)

    x_train = X[train, :]
    y_train = y[train]
    x_test = X[test, :]
    y_test = y[test]
    test_predicted = predictions

    # Plot training set
    print(y_test)
    print(test_predicted)

    coincident = 0
    for i in range(y_test.size):
        if y_test[i] == test_predicted[i]:
            coincident += 1

    print("size -> ", y_test.size)
    print("coincident -> ", coincident)
    print("Not coincident -> ", y_test.size - coincident)
    print("Error -> ", training_error)

    np.random.seed(42)

    for n_tot in range(20, 200):
        n = int(n_tot / 2)  # will use half in training, half in testing

        # two blobs; labels are 0 and 1
        X, y = make_blobs(n_tot, centers=2, cluster_std=4.0, random_state=1)

        # divide into training and testing
        order = np.random.permutation(n_tot)
        train = order[:n]  # these will be the training samples
        test = order[n:]  # and these are for testing

        my_classifier = Perceptron()
        # this is how to train the model with data in training set
        my_classifier.fit(X[train, :], y[train])
        # and this is how to predict the labels for test data
        predictions = my_classifier.predict(X[test, :])

        training_error = 1 - accuracy_score(y[test], predictions, normalize=False) / float(y[train].size)
