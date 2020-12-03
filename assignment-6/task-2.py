import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_digits
from sklearn.metrics import pairwise_distances, accuracy_score

# the constants for creating the data
n_tot = 200
ntr = 50
nts = n_tot - ntr
nc = 10

X, y = load_digits(n_class=nc, return_X_y=True)

print(y)

# divide into training and testing
Xtr = X[:ntr, :]
ytr = y[:ntr]
Xts = X[ntr:(ntr + nts), :]
yts = y[ntr:(ntr + nts)]

rcode_len = 6

# Codeword matrix
codeword_matrix = np.round(np.random.rand(nc, rcode_len))
print(codeword_matrix)

classifiers = [Perceptron() for _ in range(rcode_len)]

for classifier in classifiers:
    classifier.fit(Xtr, ytr)

for classifier in classifiers:
    print(classifier.predict([Xts[0]]))

# # the hints for coding:
# pdists = pairwise_distances(X, Y, metric="hamming") * X.shape[1]}
# np.round(np.random.rand(nc, rcode_len))
