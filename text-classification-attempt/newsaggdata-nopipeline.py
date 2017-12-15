import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

# print(twenty_train.target_names) #prints all the categories
# print("\n".join(twenty_train.data[0].split("\n")[:3])) #prints first line of the first data file

print(twenty_train.target)

# CATEGORY News category (b = business, t = science and technology, e = entertainment, m = health)
news = np.genfromtxt("./data/newsCorpora-trimmed.txt"
      ,delimiter='\t'
      ,skip_header=1
      ,dtype=None
      # ,max_rows=30
      ,names=['data','target'])

n_data = news['data'].astype(str)
temp_target = news['target'].astype(str)

target = ''
n_target = []

for x in temp_target:
    target = x.replace('b', '1')
    target = target.replace('t', '2')
    target = target.replace('e', '3')
    target = target.replace('m', '4')
    real_target = int(target)
    n_target.append(real_target)
    # print(type(real_target))
    target = ''

print(n_target)

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

X_data_counts = count_vect.fit_transform(n_data)
print(X_data_counts.shape)

X_data_tfidf = tfidf_transformer.fit_transform(X_data_counts)
print(X_data_tfidf.shape)

text_clf = MultinomialNB().fit(X_data_tfidf, n_target)

# twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
# predicted = clf.predict(twenty_test.data)
# np.mean(predicted == twenty_test.target)
