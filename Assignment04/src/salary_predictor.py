'''
salary_predictor.py
Predictor of salary from old census data -- riveting!
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *

class SalaryPredictor:

    def __init__(self, X_train, y_train):
        """
        Creates a new SalaryPredictor trained on the given features from the
        preprocessed census data to predicted salary labels. Performs and fits
        any preprocessing methods (e.g., imputing of missing features,
        discretization of continuous variables, etc.) on the inputs, and saves
        these as attributes to later transform test inputs.

        :param DataFrame X_train: Pandas DataFrame consisting of the
        sample rows of attributes pertaining to each individual
        :param DataFrame y_train: Pandas DataFrame consisting of the
        sample rows of labels pertaining to each person's salary
        """
        self.num_features = ['age', 'education_years', 'capital_gain', 'capital_loss', 'hours_per_week']
        self.cat_features = ['work_class', 'education', 'marital','occupation_code','relationship','race','sex','country','class']
        nums = X_train.filter(items=self.num_features).to_numpy()
        cats = X_train.filter(items=self.cat_features)
        # nums = X_train.select_dtypes(include='number')
        # cats = X_train.select_dtypes(include='object')


        imp_mean = SimpleImputer(missing_values='?', strategy='constant', fill_value='0.0')
        filled_list = imp_mean.fit_transform(cats)
        self.enc = preprocessing.OneHotEncoder()
        self.enc_cats = self.enc.fit(filled_list)
        self.enc_cats_arr = self.enc_cats.transform(filled_list).toarray()

        i = 0
        self.newList = []
        for list in nums:
          self.newList.append(np.concatenate((list, self.enc_cats_arr[i]), axis=None).tolist())
          i += 1
        # return newList

        self.clf = LogisticRegression(max_iter=10000).fit(self.newList, y_train)


    def classify (self, X_test):
        """
        Takes a DataFrame of rows of input attributes of census demographic
        and provides a classification for each. Note: must perform the same
        data transformations on these test rows as was done during training!

        :param DataFrame X_test: DataFrame of rows consisting of demographic
        attributes to be classified
        :return: A list of classifications, one for each input row X=x
        """
        nums = X_test.filter(items=self.num_features).to_numpy()
        cats = X_test.filter(items=self.cat_features)
        # nums = X_test.select_dtypes(include='number')
        # cats = X_test.select_dtypes(include='object')
        imp_mean = SimpleImputer(missing_values='?', strategy='constant', fill_value='0.0')
        filled_list = imp_mean.fit_transform(cats)
        enc_cats = self.enc_cats.transform(filled_list).toarray()
        i = 0
        newList = []
        for list in nums:
          newList.append(np.concatenate((list, enc_cats[i]), axis=None).tolist())
          i += 1
        return self.clf.predict(newList)

    def test_model (self, X_test, y_test):
        """
        Takes the test-set as input (2 DataFrames consisting of test demographics
        and their associated labels), classifies each, and then prints
        the classification_report on the expected vs. given labels.

        :param DataFrame X_test: Pandas DataFrame consisting of the
        sample rows of attributes pertaining to each individual
        :param DataFrame y_test: Pandas DataFrame consisting of the
        sample rows of labels pertaining to each person's salary
        """
        # [!] TODO
        print(classification_report(y_test, self.classify(X_test)))


def load_and_sanitize (data_file):
    """
    Takes a path to the raw data file (a csv spreadsheet) and
    creates a new Pandas DataFrame from it with the sanitized
    data (e.g., removing leading / trailing spaces).
    NOTE: This should *not* do the preprocessing like turning continuous
    variables into discrete ones, or performing imputation -- those
    functions are handled in the SalaryPredictor constructor, and are
    used to preprocess all incoming test data as well.

    :param string data_file: String path to the data file csv to
    load-from and fashion a DataFrame from
    :return: The sanitized Pandas DataFrame containing the demographic
    information and labels. It is assumed that for n columns, the first
    n-1 are the inputs X and the nth column are the labels y
    """
    # [!] TODO
    return pd.read_csv(data_file, encoding="latin-1", skipinitialspace=True)


if __name__ == "__main__":
    # [!] TODO
    test = load_and_sanitize('../dat/salary.csv')
    features = ['age','work_class','education','education_years','marital','occupation_code','relationship','race','sex','capital_gain','capital_loss','hours_per_week','country']
    X_train, X_test, y_train, y_test = train_test_split(test[features], test['class'], test_size=0.10)
    # print(X_train)
    # print(X_test)
    # print(y_train)
    # print(y_test)
    sp_instance = SalaryPredictor(X_train, y_train)

    # print(sp_instance.classify([[12]])
    sp_instance.test_model(X_test, y_test)