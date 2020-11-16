import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def perceptron(X, labels, sample_weight):
    # Inputs:
    # X: a 2d array, each row represents an example of the training set
    # labels: vector of the examples labels
    # sample_weight:vector of examples weights given by Adaboost
    # Output:
    # pred_labels: the label predicted for each example

    d = np.shape(X)[1]
    w = np.zeros(d)
    i = 1
    while any([element <= 0 for element in [labels[ind] * np.dot(w, x) for ind, x in enumerate(X)]]):
        # misclassified examples
        mistakes = np.where([element <= 0 for element in [labels[ind] * np.dot(w, x) for ind, x in
                                                          enumerate(X)]])[0]

        sample_weight = np.array(sample_weight)
        pairs = zip(mistakes, sample_weight[mistakes])
        sorted_pairs = sorted(pairs, key=lambda t: t[1], reverse=True)
        # use the misclassified example with maximum weight given by Adaboost
        misclass = sorted_pairs[0][0]
        # weight update
        w = w + labels[misclass] * X[misclass]
        # labels prediction
        pred_labels = [1 if x > 0 else -1 for x in [np.dot(w, x) for x in X]]
        i += 1

        if i > 201:
            break

    return pred_labels


def compute_empirical_error(weights, predicted_labels, real_labels):
    sum_weights = [weight if predicted_labels[i] != real_labels[i] else 0.0
                   for i, weight in enumerate(weights)]
    return np.sum(sum_weights)


x1 = [.1, .2, .4, .8, .8, .05, .08, .12, .33, .55, .66, .77, .22, .2, .3, .6, .5, .6, .25, .3, .5, .7, .6]
x2 = [.2, .65, .7, .6, .3, .1, .4, .66, .22, .65, .68, .55, .44, .1, .3, .4, .3, .15, .15, .5, .55, .2, .4]
labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
m = len(x1)

df = pd.DataFrame({'x1': x1, 'x2': x2, 'bias': [1.0 for i in range(m)]})

T = 5
weights = [[1.0 / m for _ in range(m)] for _ in range(T)]
alphas = []

for t in range(T):
    predicted_labels = perceptron(df.to_numpy(), labels, weights[t])
    empirical_error = compute_empirical_error(weights[t], predicted_labels, labels)

    alpha = 0.5 * np.log((1.0 - empirical_error) / empirical_error)
    alphas.append(alpha)

    z = 2.0 * np.sqrt(empirical_error * (1.0 - empirical_error))
    update = [(weights[t][i] * np.exp(-alpha * labels[i] * predicted_labels[i])) / z for i in range(m)]

    if t == T - 1:
        weights[t] = update
    else:
        weights[t + 1] = update

    print(f"Iteration {t}")
    print(weights[t])
    print("eps_t -> ", empirical_error)
    print("sum weights_t -> ", sum(weights[t]))

print(weights)

# Compute the f's
f = []

for t in range(T):
    h_t = np.array(perceptron(df.to_numpy(), labels, weights[t]))
    f.append((alphas[t] * h_t))

print("f values")
print(np.sign(np.sum(f, axis=0)))

predicted_labels = np.sign(np.sum(f, axis=0))
acc = accuracy_score(labels, predicted_labels)

# acc = compute_accuracy(df, labels, weights)
print("accuracy -> ", acc)
