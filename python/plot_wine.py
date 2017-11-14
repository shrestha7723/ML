import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

data = np.loadtxt("../data/wine.data",delimiter=",",usecols=(1,2))
target = np.loadtxt("../data/wine.data",delimiter=",",usecols=0)

clf = svm.SVC(kernel='linear',C=1.0)

clf.fit(data,target)

print(clf)
