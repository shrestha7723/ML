# Trying MLP with auto-mpg data

import numpy as np

training_data = np.genfromtxt("../data/auto-mpg.csv"
      ,dtype={'names': ('mpg', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'type')
             ,'formats': ('<f8', '<i4', '<f8', '<f8', '<f8', '<f4', '<i4', '<i2','|S32')}
      ,missing_values='?'
      ,delimiter=',')

target = training_data[:][:1]

target = np.ndarray(target)
# inputs = np.ndarray(training_data[:][1:8])

learning_rate = 0.3

print(target)
# print(inputs)
