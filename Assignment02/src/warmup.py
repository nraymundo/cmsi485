import numpy as np
import pandas as pd
from pomegranate import *

'''
warmup.py

Skeleton for answering warmup questions related to the
AdAgent assignment. By the end of this section, you should
be familiar with:
- Importing, selecting, and manipulating data using Pandas
- Creating and Querying a Bayesian Network
- Using Samples from a Bayesian Network for Approximate Inference

@author: Andrew Rossell, Damián Browne, Nico Raymundo
'''

if __name__ == '__main__':
    """
    PROBLEM 2.1
    Using the Pomegranate Interface, determine the answers to the
    queries specified in the instructions.
    """

    # TODO: 2.1
    def get_proba(Q, e = {}):
        """
        Helper function that returns the probability for the query given the evidence
        :param string Q: represent the query variable
        :param dict e: dictionary with all of the evidence variables and their values
        """
        model_proba = model.predict_proba(e)
        return model_proba[f.columns.get_loc(Q)].parameters[0]

    f = pd.read_csv('../dat/adbot-data.csv', sep=',')
    state_names = f.columns
    model = BayesianNetwork.from_samples(X = f, state_names = state_names, algorithm = 'exact')

    print('P(S): ', get_proba('S'))
    """
    P(S):
      0 : 0.5455778344088269
      1 : 0.2433755611884188
      2 : 0.21104660440275425
    """
    print('P(S|G=1): ', get_proba('S', {'G':1}))
    """
    P(S|G=1):
      0 : 0.46958553823567634
      1 : 0.2699449056011039
      2 : 0.2604695561632199
    """
    print('P(S|T=1, H=1): ', get_proba('S', {'T':1, 'H':1}))
    """
    P(S|T=1,H=1):
      0 : 0.403472778260472
      1 : 0.3073574386129309
      2 : 0.2891697831265971
    """