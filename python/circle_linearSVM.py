import numpy as np

from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification

data = np.genfromtxt("../data/circle.csv"
      ,skip_header=1
      ,delimiter=','
      ,names=['X','y','class'])

X = np.stack((data['X'],data['y']))

print(data['X'].shape)

# X, y = make_classification(n_features=4, random_state=0)

# clf = LinearSVC(random_state=0)
# clf.fit(X, y)
#
# print(clf.coef_)
#
# print(clf.intercept_)
