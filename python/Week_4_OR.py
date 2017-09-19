import random as rand
import numpy as np

#Perceptron OR

def sgn(input):
    if input > 0:
        return 1
    else:
        return 0

#[x1,x2,t]
training_data = np.array([
[0,0,0],
[1,0,1],
[0,1,1],
[1,1,1],
])

w1 = rand.randrange(2)
w2 = rand.randrange(2)

print("W1 = {}".format(w1))
print("W2 = {}".format(w2))

weight = np.array([w1, w2])

learning_rate = 0.5

for j in range(5):
    flag = 0
    for x in training_data:
        row = x[:2]
        target = x[2:][0]

        output = np.dot(row,weight)

        output = sgn(output)

        d_weight1 = learning_rate * (target - output) * row[0]
        d_weight2 = learning_rate * (target - output) * row[1]

        d_weights = np.array([d_weight1, d_weight2])

        #print("W  = {}".format(weight))
        #print("DW = {}".format(d_weights))

        weight = weight + d_weights

        #print("WW = {}".format(weight))

        erot = target - output

        if erot != 0:
            flag = 1

        print("T - O = {}".format(target - output))

    if flag == 0:
        print("Done after {} loop".format(j+1))
        break
