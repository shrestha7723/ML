
# Using iris data

from sklearn import datasets
import numpy as np

#Multilayer percetron v.0.1

def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

def derivate_output(output):
    return output*(1-output)
# Create array with n numbers of normalized random weight
def random_weight():
    return 2*np.random.random((num_inputs,))-1

iris = datasets.load_iris(True)
data, target = iris

learning_rate = 0.2
num_interates = 0

num_inputs = 4
num_layer = 3
num_inodes = 3
num_hnodes = 1
num_onodes = 3

np.random.seed(1)
#Node in each layer as an array
#Each node contains n amount of weights
ilayer = [random_weight() for iter in range(num_inodes)]
hlayer = [random_weight() for iter in range(num_hnodes)]
olayer = [random_weight() for iter in range(num_onodes)]
print(ilayer)
print(hlayer)
print(olayer)

for iter in range(num_interates):

    output_ilayer = [sigmoid(np.dot(data, ilayer[i])) for i in range(num_inodes)]
    output_hlayer = [sigmoid(np.dot(output_ilayer, hlayer[i])) for i in range(num_inodes)]
    output_olayer = [sigmoid(np.dot(output_hlayer, olayer[i])) for i in range(num_inodes)]
    print(output_olayer[0])
        #o_inode = sigmoid(np.dot(data, inode))
        #o_hidden = sigmoid(np.dot(o_inode, hnode))
        #output = sigmoid(np.dot(o_hidden, onode))
        #output_error = target - output
        #output_delta = learning_rate * output_error * o_hidden * derivate_output(output)
