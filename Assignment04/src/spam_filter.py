'''
spam_filter.py
Spam v. Ham Classifier trained and deployable upon short
phone text messages.
'''
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import *

class SpamFilter:

    def __init__(self, text_train, labels_train):
        """
        Creates a new text-message SpamFilter trained on the given text
        messages and their associated labels. Performs any necessary
        preprocessing before training the SpamFilter's Naive Bayes Classifier.
        As part of this process, trains and stores the CountVectorizer used
        in the feature extraction process.

        :param DataFrame text_train: Pandas DataFrame consisting of the
        sample rows of text messages
        :param DataFrame labels_train: Pandas DataFrame consisting of the
        sample rows of labels pertaining to each text message
        """
        self.vectorizer = CountVectorizer(stop_words='english')
        self.features = self.vectorizer.fit_transform(text_train)

        self.nbc = MultinomialNB().fit(self.features, labels_train)

    def classify (self, text_test):
        """
        Takes as input a list of raw text-messages, uses the SpamFilter's
        vectorizer to convert these into the known bag of words, and then
        returns a list of classifications, one for each input text

        :param list/DataFrame text_test: A list of text-messages (strings) consisting
        of the messages the SpamFilter must classify as spam or ham
        :return: A list of classifications, one for each input text message
        where index in the output classes corresponds to index of the input text.
        """
        test_features = self.vectorizer.transform(text_test)
        return self.nbc.predict(test_features)

    def test_model (self, text_test, labels_test):
        """
        Takes the test-set as input (2 DataFrames consisting of test texts
        and their associated labels), classifies each text, and then prints
        the classification_report on the expected vs. given labels.

        :param DataFrame text_test: Pandas DataFrame consisting of the
        test rows of text messages
        :param DataFrame labels_test: Pandas DataFrame consisting of the
        test rows of labels pertaining to each text message
        """
        print(classification_report(labels_test, self.classify(text_test)))


def load_and_sanitize (data_file):
    """
    Takes a path to the raw data file (a csv spreadsheet) and
    creates a new Pandas DataFrame from it with only the message
    texts and labels as the remaining columns.

    :param string data_file: String path to the data file csv to
    load-from and fashion a DataFrame from
    :return: The sanitized Pandas DataFrame containing the texts
    and labels
    """
    data = pd.read_csv(data_file, encoding="latin-1")
    data = data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
    data = data.rename(columns={"v1": "class", "v2": "text"})
    return data

file = '../dat/texts.csv'

if __name__ == "__main__":
    test = load_and_sanitize(file)
    X_train, X_test, y_train, y_test = train_test_split(test['text'], test['class'], test_size=0.10)
    sf_instance = SpamFilter(X_train, y_train)
    sf_instance.test_model(X_test, y_test)