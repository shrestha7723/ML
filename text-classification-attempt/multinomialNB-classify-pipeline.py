import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def main():
    # Limiting the categories loaded
    # categories = ['alt.atheism', 'soc.religion.christian',
    #               'comp.graphics', 'sci.med']

    twenty_train = fetch_20newsgroups(
        subset='train', shuffle=True)
    # twenty_train = fetch_20newsgroups(
    #     subset='train', categories=categories, shuffle=True)

    # All target target name
    # print(twenty_train.target_names)

    # print("\n".join(twenty_train.data[0].split("\n")[:3]))

    # First data and its target value
    # print(twenty_train.data[0])
    # print(twenty_train.target_names[twenty_train.target[0]])

    text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
                         ])

    text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

    twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
    # twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True)
    predicted = text_clf.predict(twenty_test.data)
    mean = np.mean(predicted == twenty_test.target)

    print(mean)

    docs_new = ['God is love', 'Spread the hate, cause the goverment is bad',
                'George bush is my neighbor']

    predicted = text_clf.predict(docs_new)

    for doc, category in zip(docs_new, predicted):
        print('%r => %s' % (doc, twenty_train.target_names[category]))

    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf__alpha': (1e-2, 1e-3),
                  }

    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)

    print(gs_clf.best_score_)
    print(gs_clf.best_params_)


if __name__ == '__main__':
    main()
