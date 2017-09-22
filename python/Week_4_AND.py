
import random as rand
import numpy as np

# importing sys for taking argument
import sys


def sgn(value):
    if value > 0:
        return 1
    else:
        return 0


def predict(inputs):
    result = np.dot(inputs, weights)

    result = sgn(result)

    print("The predicted output is {}".format(result))


def ask_input():
    inputs = np.array([0,0,0])

    for x in range(3):
        inputs[x] = input("Enter input {} = ".format(x))

    return inputs


training_data = np.array([
[0,0,0,0],
[0,0,1,0],
[0,1,0,0],
[0,1,1,0],
[1,0,0,0],
[1,1,1,0],
[1,1,0,0],
[1,1,1,1]
])

w1 = rand.randrange(1)
w2 = rand.randrange(1)
w3 = rand.randrange(1)

weights = np.array([w1, w2, w3])

learning_rate = 0.5

print("W1 = {}".format(w1))
print("W2 = {}".format(w2))
print("W3 = {}".format(w3))

for j in range(100):
    flag = 0
    for x in training_data:
        row = x[:3]
        target = x[3:][0]

        output = np.array([0,0,0])

        for y in range(3):
            output[y] = row[y] * weights[y]

        output = np.sum(output)

        #output = np.dot(row,weights)

        output = sgn(output)

        d_weights = np.array([0,0,0])

        for k in range(len(row)):
            d_weights[k] = learning_rate * (target - output) * row[k]

        erot = target - output
        print("T - O = {}".format(erot))

        weights = weights + d_weights

        if erot != 0:
            flag = 1
            #print("Looping...")

    if flag == 0:
        print("Done after {} loop".format(j+1))
        break

if len(sys.argv) > 1:
    if sys.argv[1] == 'd':
        predict(ask_input())

    if sys.argv[1] == 'i':

        inputs = [int(i) for i in sys.argv[2:5]]
        predict(inputs)
