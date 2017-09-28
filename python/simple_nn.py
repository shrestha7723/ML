# source: http://iamtrask.github.io (modified)
import numpy as np

X_XOR = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
y_truth = np.array([[0,1,1,1]]).T
np.random.seed(1)
synapse_0 = 2*np.random.random((3,1)) - 1

def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output
def sigmoid_output_to_derivative(output):
    return output*(1-output)

for iter in range(10000):
    layer_1 = sigmoid(np.dot(X_XOR, synapse_0))
    layer_1_error = layer_1 - y_truth
    layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
    synapse_0_derivative = np.dot(X_XOR.T,layer_1_delta)
    synapse_0 -= synapse_0_derivative

print("Output After Training: \n", layer_1)
