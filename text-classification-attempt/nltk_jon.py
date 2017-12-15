import nltk
from nltk.corpus import brown
import numpy as np
import string
import csv

training = []
kata = []
all_words = []
train_set = []
test_set = []
with open('topic-cropped.csv', 'r', encoding="UTF-8") as news:
    reader = csv.reader(news)
    for row in reader:
        print(row)
        training.append(row)
        kata.append(row[0])

    length = len(training)
    print(length)

    for w in kata:
        all_words.append(w.lower())

    all_words = nltk.FreqDist(all_words)
    word_features = list(all_words.keys())[:3000]

    #trainingSetLength = length - 1000
    #testSetLength = length - 1226

    # for row in range (0,trainingSetLength):
    #     train_set.append(training[row])

    # for row in range(trainingSetLength+1,length):
    #     test_set.append(training[row])

    # print(train_set)
    def find_features(training):
        words = set(training)
        features = {}
        for w in word_features:
            features[w] = (w in words)

        return features

    featuresets = [(find_features(text), category)
                   for (text, category) in training]
    train_set = featuresets[0:1500]
    test_set = featuresets[1400:1500]
    # print(train_set)
    # print(test_set)

    classifier = nltk.NaiveBayesClassifier.train(train_set)
    #coba = [["podcasters look to net money", "false"]]

    print("Classifier accuracy percent:",
          (nltk.classify.accuracy(classifier, test_set)) * 100)
    # print(trainingSet)
    # print(testSet)

    # Training

# new_teks= string.replace(teks,' ','')
# print(new_teks)

# brown.categories())

# tagged_sents = list(brown.sents(categories='news'))

# print(tagged_sents)

# news_text = brown.words(categories='news')
# fdist = nltk.FreqDist(w.lower() for w in news_text)
# modals = ['can', 'could', 'may', 'might', 'must', 'will']
# for m in modals:
#     print(m + ':', fdist[m], end=' ')

# # print(classifier = nltk.NaiveBayesClassifier.train(tagged_sents))
