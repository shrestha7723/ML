
import random as rand
import numpy as np

def sgn(input):
    if input > 0:
        return 1
    else:
        return 0

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

for j in range(10):
    flag = 0
    for x in training_data:
        row = x[:3]
        target = x[3:][0]

        output = np.dot(row,weights)

        output = sgn(output)

        #print("Aut = {}".format(output))

        d_weight1 = learning_rate * (target - output) * row[0]
        d_weight2 = learning_rate * (target - output) * row[1]
        d_weight3 = learning_rate * (target - output) * row[2]

        d_weights = np.array([d_weight1,d_weight2,d_weight3])
        erot = target - output
        print("T - O = {}".format(erot))

        weights = weights + d_weights

        if erot != 0:
            flag = 1
            #print("Looping...")
    print("Done after {} loop".format(j+1))
    break
