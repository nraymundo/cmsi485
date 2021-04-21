'''
salary_predictor_grading_demo.py
Small test to ensure that your SalaryPredictor works for what will
be the grading tests
'''
import pandas as pd
from salary_predictor import *

if __name__ == "__main__":
    training_data = load_and_sanitize("../dat/salary.csv")
    X_train = training_data[training_data.columns[0:training_data.columns.size-1]]
    y_train = training_data[training_data.columns[-1]]
    salary = SalaryPredictor(X_train, y_train)
    
    # [!] NOTE: The below will load a separate grading set of samples during grading;
    # the following should run without errors for you to have a gradeable submission
    test_data = load_and_sanitize("../dat/salary.csv")
    X_test = test_data[test_data.columns[0:test_data.columns.size-1]]
    y_test = test_data[test_data.columns[-1]]
    salary.test_model(X_test, y_test)
    