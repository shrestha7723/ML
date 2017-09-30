# Multilayer percetron v.0.3
# Using iris data

from sklearn import datasets
import numpy as np


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


def derivate_output(output):
    return output * (1 - output)


# Create array with n numbers of normalized random weight
def random_weight(size):
    return 2*np.random.random((size,))-1

iris = datasets.load_iris(True)
data, target = iris
target = target.reshape((150, 1))

# print("Iris")
# print(data.shape)
# print(target.shape)

learning_rate = 0.2
num_interates = 2

# Let's try 3-1-1 for iris data
num_inputs = 4
num_layer = 3
num_inodes = 3
num_hnodes = 1
num_onodes = 1

np.random.seed(1)
# Node in each layer as an array
# Each node contains n amount of weights
ilayer = np.array([random_weight(num_inputs) for iter in range(num_inodes)])
hlayer = np.array([random_weight(num_inodes) for iter in range(num_hnodes)])
olayer = np.array([random_weight(num_hnodes) for iter in range(num_onodes)])
print("Layer")
print(ilayer.shape)
# print(hlayer.shape)
# print(olayer.shape)

for iter in range(num_interates):

    # Calculate the output of the network starting from input layer
    output_ilayer = np.array([sigmoid(np.dot(data, ilayer[i]))
                              for i in range(num_inodes)]).transpose()
    # print("Output")
    # print(output_ilayer.shape)
    output_hlayer = np.array([sigmoid(np.dot(output_ilayer, hlayer[i]))
                              for i in range(num_hnodes)]).transpose()
    # print(output_hlayer.shape)
    output_olayer = np.array([sigmoid(np.dot(output_hlayer, olayer[i]))
                              for i in range(num_onodes)]).transpose()
    # print(output_olayer.shape)

    # Calculate the delta weights of the output layer
    err_olayer = np.array([target - output_olayer[i] for i in range(num_onodes)])
    delta_w_output = np.array([learning_rate * err_olayer[i]
                      * derivate_output(output_olayer[i]) * output_hlayer for i in range(num_onodes)])
    total_delta_w_output = np.sum(delta_w_output, axis=0)

    # Calculate the delta weights of the hidden layer
    sum_downstream = np.array([derivate_output(output_olayer) * olayer[i] for i in range(num_onodes)])
    print(sum_downstream.shape)
    delta_hidden = derivate_output(output_hlayer) * sum_downstream
    delta_w_hidden = learning_rate * delta_hidden * output_ilayer
    total_delta_w_hidden = np.sum(delta_w_hidden, axis=0)

    # Calculate the delta weights of the input layer
    sum_downstream = derivate_output(output_hlayer) * hlayer
    delta_input = derivate_output(output_ilayer) * sum_downstream
    print("Hai")
    print(delta_input[0].shape)

    delta_w_input = np.array([learning_rate * delta_input[i]
                     * data for i in range(num_inodes)])
    print(delta_w_input.shape)
    total_delta_w_input = np.sum(delta_w_input, axis=0)

    # Update all weights
    olayer = olayer - total_delta_w_output
    hlayer = hlayer - total_delta_w_hidden
    ilayer = ilayer - total_delta_w_input
