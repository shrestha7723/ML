#Testing out genfromtxt and new data

import numpy as np

decode = lambda s: s.decode(encoding='UTF-8')

data = np.genfromtxt("../data/auto-mpg.csv"
      ,dtype={'names': ('mpg', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'type')
             ,'formats': ('<f8', '<i4', '<f8', '<f8', '<f8', '<f4', '<i4', '<i2','|S32')}
      ,missing_values='?'
      ,delimiter=','
      ,converters={8: decode})

# print(data[1])
# print(type(data[1]))

print(data[1][8])
print(type(data[1]['type'].decode(encoding='UTF-8')))
