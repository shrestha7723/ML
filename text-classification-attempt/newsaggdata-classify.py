import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import pandas as pd

def strip(text):
    try:
        return text.strip()
    except AttributeError:
        return text

# news = pd.read_csv('./data/newsCorpora.csv'
#                  ,sep='\t'
#                  ,header=None
#                  ,usecols=[1,4]
#                  # ,dtype=str
#                  ,nrows=100
#                  ,names=['data','target']
#                  # ,converters={'data': strip}
#                  )

news = np.genfromtxt("./data/newsCorpora-trimmed.txt"
      ,delimiter='\t'
      ,skip_header=1
      ,dtype=None
      ,max_rows=30
      ,names=['data','target'])

n_data = news['data'].astype(str)
n_target = news['target'].astype(str)


print(n_data)
print(n_target)

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])

text_clf = text_clf.fit(n_data, n_target)

# docs_new = ['God is love', 'Spread the hate, cause the goverment is bad', 'George bush is my neighbor']

# predicted = text_clf.predict(docs_new)
#
# for doc, category in zip(docs_new, predicted):
#     print('%r => %s' % (doc, twenty_train.target_names[category]))

#
# parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
#               'tfidf__use_idf': (True, False),
#               'clf__alpha': (1e-2, 1e-3),
#               }
#
# gs_clf = GridSearchCV(text_clf, parameters, n_jobs=1)
# gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)
#
# print(gs_clf.best_score_)
# print(gs_clf.best_params_)
