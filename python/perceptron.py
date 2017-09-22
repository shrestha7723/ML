
# Single perceptron
# source from floydhub

from random import choice
from numpy import array, dot, random
signum = lambda x: 0 if x < 0 else 1
# Traning data as an array of tupple instead of array of array
# for easier separation of input and target value
training_data = [ (array([0,0,1]), 0),
                    (array([0,1,1]), 1),
                    (array([1,0,1]), 1),
                    (array([1,1,1]), 1), ]
# rand(d0, d1, ... , dn) return an array of random values
# shaped with the dimension parameter
# rand(3,2) -> [[0, 1],[1, 1],[0, 1]]
# rand(5) -> [0, 1, 0.5, 0, 1] or smth like that, seems like
# the values are between 0 and 1
weights = random.rand(3)
errors = []
# 0.5 learning_rate isn't good enough...
learning_rate = 0.2
num_iterations = 100

# da learning process
for i in range(num_iterations):
    # choice(collection) selects one of the obj. in the collection randomly
    # continued with basic tupple assignment
    input, truth = choice(training_data)
    result = dot(weights, input)
    error = truth - signum(result)
    errors.append(error)
    # accumulated the entire delta weight
    weights += learning_rate * error * input

# print out the predicted value using the updated weights
for x, _ in training_data:
    result = dot(x, weights)
    print("{}: {} -> {}".format(x[:3], result, signum(result)))
