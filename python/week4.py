
import random as rand
import numpy as np

def err(input, ):
    pass

def sgn(input):
    if input > 0:
        return 1
    else:
        return -1

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

weight = np.array([w1, w2, w3])

learning_rate = 0.5

error_threshold = 0.0

print("W1 = {}".format(w1))
print("W2 = {}".format(w2))
print("W3 = {}".format(w3))

flag = 1

counter = 1

while flag == 1:
    flag = 0
    for x in training_data:
        row = x[:3]
        target = x[3:][0]

        output = np.dot(row,weight)

        output = sgn(output)

        #print("Aut = {}".format(output))

        dewe1 = learning_rate * (target - output) * row[0]
        dewe2 = learning_rate * (target - output) * row[1]
        dewe3 = learning_rate * (target - output) * row[2]

        dewe = np.array([dewe1,dewe2,dewe3])

        sum_dewe = np.sum(dewe)
        #print("hai")
        #print(sum_dewe)

        #print("Target = {}".format(target))
        #print("Output = {}".format(output))

        print("T - O = {}".format(target - output))

        weight = weight + dewe

        if sum_dewe != 0:
            flag = 1
            #print("Looping...")
    print("Weight ke-{} = {}".format(counter,weight))
    counter = counter + 1

print("\n")
