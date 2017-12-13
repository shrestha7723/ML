import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

# print(twenty_train.target_names) #prints all the categories
# print("\n".join(twenty_train.data[0].split("\n")[:3])) #prints first line of the first data file

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

X_train_counts = count_vect.fit_transform(twenty_train.data)
print(X_train_counts.shape)

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
predicted = clf.predict(twenty_test.data)
np.mean(predicted == twenty_test.target)
