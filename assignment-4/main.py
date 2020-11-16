import numpy as np

# Load the data
m0 = 100  # m0+1 gives the index of the 0 input
m = 2 * m0 + 1  # number of points
xx = 2 * np.arange(m + 1) / m - 1  # input data is in the range [-1,+1], with steps 0.01
rr = xx ** 2  # output values: a parabola

# set the learner parameters
n_iteration = 100  # number of iterations
H = 5  # number of nodes in the hidden layer
eta = 0.3  # learning speed, step size

# Random initialization of the edge weights
# Set the seed to a fixed number
np.random.seed(12345)

# weights between the input and the hidden layers
W = 2 * (np.random.rand(H) - 0.5) / 100  # random uniform in [-0.01,0.01]
# weights between the hidden and the output layers
V = 2 * (np.random.rand(H) - 0.5) / 100  # random uniform in [-0.01,0.01]

print("Input data -> ", xx)
print("Input data length -> ", xx.shape[0])
print(f"W = {W}")
print(f"V = {V}")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


random_index = 0

for i in range(n_iteration):
    x = xx[random_index]
    r = rr[random_index]
    random_index += 1

    # Calculation of the output of intermediate layer (using sigmoid activation)
    z = sigmoid(-W.T * x)
    # Calculation of the output of the net
    y = np.inner(V.T, z)

    # calculation of deltas
    error = r - y

    delta_v = eta * error * z
    delta_w = eta * error * V * ((z * (1 - z)) * x)

    # delta_w_2 = np.array([(eta * (r - y) * V[h] * z[h] * (1 - z[h]) * x)[0] for h in range(H)])

    V = V + delta_v
    W = W + delta_w

    print(f"({x}, {r})")
    print(f"z = {z}")
    print(f"y = {y}")
    print(f"delta v = {delta_v}")
    print(f"delta w = {delta_w}")
    print(f"W = {W}")

# Compute the predicted values
yy = np.array([])

for i in range(m):
    x = xx[i]
    wh = W * x
    # Calculation of the output of intermediate layer (using sigmoid activation)
    z = 1.0 / (1 + np.exp(-W * x))
    # Calculation of the output of the net
    y = np.inner(V, z)
    yy = np.append(yy, y)

print(f"len r -> {rr.shape[0]}")
print(f"len y -> {yy.shape[0]}")

error = rr[m0 + 1] - yy[m0 + 1]
print(f"E = {error}")
