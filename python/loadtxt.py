#Testing out loadtxt and new data

import numpy as np

data = np.genfromtxt("../data/auto-mpg.csv"
      ,dtype={'names': ('x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8','type')
             ,'formats': ('<f4', '<i4', '<f8', '<f4', '<f4', '<f4', '<i4', '<i2','|S32')}
      ,missing_values='?'
      ,delimiter=',')

print(data[0])
