
import random as rand
import numpy as np

# importing sys for taking argument
import sys


def signum(x): return 0 if x < 0 else 1


def predict(inputs, target):
    result = np.dot(inputs, weights)

    result = signum(result)

    print("Target = {}, predicted = {}".format(target, result))


def random_weight(size):
    return 2 * np.random.random((size,)) - 1


training_data = np.array([
    [0, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 1, 0],
    [1, 1, 1, 1]
])
np.random.seed(1)

weights = random_weight(3)

learning_rate = 0.3

print("W = {}".format(weights))

for j in range(500):

    x = rand.choice(training_data)

    row = x[:3]
    target = x[3:][0]

    output = [0, 0, 0]

    output = np.dot(row, weights)

    output = signum(output)

    # d_weights = np.array([0,0,0])
    d_weights = [0, 0, 0]

    for k in range(len(row)):
        d_weights[k] = learning_rate * (target - output) * row[k]

    # print("dW = {}".format(d_weights))

    weights = weights + d_weights
    print("W = {}".format(weights))


if len(sys.argv) > 1:
    if sys.argv[1] == 't':
        for i in range(len(training_data)):
            predict(training_data[i][:3], training_data[i][3:][0])

    if sys.argv[1] == 'i':
        inputs = [int(i) for i in sys.argv[2:5]]
        predict(inputs)
