#Testing out loadtxt and new data

import numpy as np

data = np.loadtxt("../data/iris.data",dtype={'names': ('x1', 'x2', 'x3', 'x4','type')
                                             ,'formats': ('f4', 'f4', 'f4', 'f4','S24')}
                  ,delimiter=',')

print(data[0])
