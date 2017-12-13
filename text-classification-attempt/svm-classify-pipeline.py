import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

# text_clf_svm = Pipeline([('vect', CountVectorizer()),
#                          ('tfidf', TfidfTransformer()),
#                          ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
#                                                    alpha=1e-3, n_iter=5, random_state=42)),
#                          ])
text_clf_svm = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                                   alpha=1e-3, max_iter=1000, random_state=42)),
                         ])

_ = text_clf_svm.fit(twenty_train.data, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
predicted_svm = text_clf_svm.predict(twenty_test.data)
mean = np.mean(predicted_svm == twenty_test.target)

print(mean)
