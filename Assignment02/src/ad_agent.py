'''
ad_engine.py
Advertisement Selection Engine that employs a Decision Network
to Maximize Expected Utility associated with different Decision
variables in a stochastic reasoning environment.

@author: Andrew Rossell, Damián Browne, Nico Raymundo
'''
import pandas as pd
from pomegranate import *
import math
import itertools
import unittest
import copy

class AdEngine:

    def __init__(self, data_file, dec_vars, util_map):
        """
        Responsible for initializing the Decision Network of the
        AdEngine using the following inputs

        :param string data_file: path to csv file containing data on which
        the network's parameters are to be learned
        :param list dec_vars: list of string names of variables to be
        considered decision variables for the agent. Example:
          ["Ad1", "Ad2"]
        :param dict util_map: discrete, tabular, utility map whose keys
        are variables in network that are parents of a utility node, and
        values are dictionaries mapping that variable's values to a utility
        score, for example:
          {
            "X": {0: 20, 1: -10}
          }
        represents a utility node with single parent X whose value of 0
        has a utility score of 20, and value 1 has a utility score of -10
        """
        self.data_file = data_file
        self.dec_vars = dec_vars                    # decision nodes
        self.util_map = util_map                    # Parents of utility node + their values
        self.data = pd.read_csv(data_file, sep=',')
        state_names = self.data.columns
        self.bn = BayesianNetwork.from_samples(X = self.data, state_names = state_names, algorithm = 'exact')

    def get_prob(self, Q, e = {}):
        """
        Helper function that returns the probability for the query given the evidence
        
        :param string Q: represent the query variable
        :param dict e: dictionary with all of the evidence variables and their values
        :return: dict with the probability values
        """
        model_prob = self.bn.predict_proba(e)
        return model_prob[self.data.columns.get_loc(Q)].parameters[0]

    def decision_combos(self):
        d_dict = dict()
        for var in self.dec_vars:
            d_dict[var] = list(self.data[var].unique())
        d_val_combos = list(itertools.product(*d_dict.values(), repeat=len(self.dec_vars)))
        d_combos = list()
        for val_combo in d_val_combos:
            i = 0
            combo = dict()
            for var in self.dec_vars:
                combo[var] = val_combo[i]
                i += 1
            d_combos.append(combo)
        return d_combos

    def meu(self, evidence):
        """
        Given some observed demographic "evidence" about a potential
        consumer, selects the ad content that maximizes expected utility
        and returns a dictionary over any decision variables and their
        best values plus the MEU from making this selection.

        :param dict evidence: dict mapping network variables to their
        observed values, of the format: {"Obs1": val1, "Obs2": val2, ...}
        :return: 2-Tuple of the format (a*, MEU) where:
          - a* = dict of format: {"DecVar1": val1, "DecVar2": val2, ...}
          - MEU = float representing the EU(a* | evidence)
        """
        # s = parents of Utility node that are chance nodes
        # a = decision/action Nodes
        # e = evidence
        # eus = list of possible EUs: summation for all s in S of P(s|a,e)*U(s)
        # ae = dict with both a and e
        d_combos = self.decision_combos()
        eus = list()
        for a in d_combos:
            summation = 0
            ae = copy.deepcopy(a)
            ae.update(evidence)
            for s in self.util_map:
                p_dist = self.get_prob(s,ae)
                for p in p_dist:
                    summation += p_dist[p] * self.util_map[s][p]
            eus.append((a, summation))
        a_star = dict()
        meu = 0
        for eu in eus:
            a_star = dict(eu[0]) if eu[1] > meu else a_star
            meu = eu[1] if eu[1] > meu else meu
        return (a_star, meu)

    def vpi(self, potential_evidence, observed_evidence):
        """
        Given some observed demographic "evidence" about a potential
        consumer, returns the Value of Perfect Information (VPI)
        that would be received on the given "potential" evidence about
        that consumer.

        :param string potential_evidence: string representing the variable name
        of the variable under evaluation
        :param dict e: dict mapping network variables
        to their observed values, of the format: {"Obs1": val1, "Obs2": val2, ...}
        :return: float value indicating the VPI(potential | observed)
        """
        observed = self.meu(observed_evidence)[1]
        summation = 0
        p_dist_ep_e = self.get_prob(potential_evidence, observed_evidence)
        for ep_val in list(self.data[potential_evidence].unique()):
            # Damián Browne made these variable names   
            e_prime = {potential_evidence: ep_val}
            ee_prime = copy.deepcopy(observed_evidence)                     # dict with both observed evidence and e_prime
            ee_prime.update(e_prime)                                        
            summation += p_dist_ep_e[ep_val] * self.meu(ee_prime)[1]
        return max(0, summation - observed)


class AdAgentTests(unittest.TestCase):

    def test_meu_lecture_example_no_evidence(self):
        ad_engine = AdEngine('../dat/lecture5-2-data.csv', ["D"], {"Y": {0: 3, 1: 1}})
        evidence = {}
        decision = ad_engine.meu(evidence)
        self.assertEqual({"D": 0}, decision[0])
        self.assertAlmostEqual(2, decision[1], delta=0.01)

    def test_meu_lecture_example_with_evidence(self):
        ad_engine = AdEngine('../dat/lecture5-2-data.csv', ["D"], {"Y": {0: 3, 1: 1}})
        evidence = {"X": 0}
        decision = ad_engine.meu(evidence)
        self.assertEqual({"D": 1}, decision[0])
        self.assertAlmostEqual(2, decision[1], delta=0.01)

        evidence2 = {"X": 1}
        decision2 = ad_engine.meu(evidence2)
        self.assertEqual({"D": 0}, decision2[0])
        self.assertAlmostEqual(2.4, decision2[1], delta=0.01)

    def test_vpi_lecture_example_no_evidence(self):
        ad_engine = AdEngine('../dat/lecture5-2-data.csv', ["D"], {"Y": {0: 3, 1: 1}})
        evidence = {}
        vpi = ad_engine.vpi("X", evidence)
        self.assertAlmostEqual(0.24, vpi, delta=0.1)

    def test_meu_defendotron_no_evidence(self):
        ad_engine = AdEngine('../dat/adbot-data.csv', ["Ad1", "Ad2"], {"S": {0: 0, 1: 1776, 2: 500}})
        evidence = {}
        decision = ad_engine.meu(evidence)
        self.assertEqual({"Ad1": 1, "Ad2": 0}, decision[0])
        self.assertAlmostEqual(746.72, decision[1], delta=0.01)

    def test_meu_defendotron_with_evidence(self):
        ad_engine = AdEngine('../dat/adbot-data.csv', ["Ad1", "Ad2"], {"S": {0: 0, 1: 1776, 2: 500}})
        evidence = {"T": 1}
        decision = ad_engine.meu(evidence)
        self.assertEqual({"Ad1": 1, "Ad2": 1}, decision[0])
        self.assertAlmostEqual(720.73, decision[1], delta=0.01)

        evidence2 = {"T": 0, "G": 0}
        decision2 = ad_engine.meu(evidence2)
        self.assertEqual({"Ad1": 0, "Ad2": 0}, decision2[0])
        self.assertAlmostEqual(796.82, decision2[1], delta=0.01)

    def test_vpi_defendotron_no_evidence(self):
        ad_engine = AdEngine('../dat/adbot-data.csv', ["Ad1", "Ad2"], {"S": {0: 0, 1: 1776, 2: 500}})
        evidence = {}
        vpi = ad_engine.vpi("G", evidence)
        self.assertAlmostEqual(20.77, vpi, delta=0.1)

        vpi2 = ad_engine.vpi("F", evidence)
        self.assertAlmostEqual(0, vpi2, delta=0.1)

    def test_vpi_defendotron_with_evidence(self):
        ad_engine = AdEngine('../dat/adbot-data.csv', ["Ad1", "Ad2"], {"S": {0: 0, 1: 1776, 2: 500}})
        evidence = {"T": 0}
        vpi = ad_engine.vpi("G", evidence)
        self.assertAlmostEqual(25.49, vpi, delta=0.1)

        evidence2 = {"G": 1}
        vpi2 = ad_engine.vpi("P", evidence2)
        self.assertAlmostEqual(0, vpi2, delta=0.1)

        evidence3 = {"H": 0, "T": 1, "P": 0}
        vpi3 = ad_engine.vpi("G", evidence3)
        self.assertAlmostEqual(66.76, vpi3, delta=0.1)

    def test_no_evidence(self):
        ad_engine = AdEngine('../dat/test.csv', ['D','Y','X','Z'], {"F": {0: 0, 1: 1776, 2: 500}})
        evidence = {}
        decision = ad_engine.meu(evidence)
        self.assertEqual({'D': 0, 'Y': 1, 'X': 2, 'Z': 0}, decision[0])
        self.assertAlmostEqual(1776, decision[1], delta=0.01)

    def test_vpi_fb(self):
        ad_engine = AdEngine('../dat/adbot-data.csv', ["Ad1", "Ad2"], {"S": {0: 0, 1: 1776, 2: 500}})
        evidence = {}
        vpi = ad_engine.vpi("G", evidence)
        self.assertAlmostEqual(20.78, vpi, delta=0.1)
    
    def test_vpi_google(self):
        ad_engine = AdEngine('../dat/adbot-data.csv', ["Ad1", "Ad2"], {"S": {0: 0, 1: 1776, 2: 500}})
        evidence = {"G": 1}
        vpi = ad_engine.vpi("P", evidence)
        self.assertAlmostEqual(0, vpi, delta=0.1)

if __name__ == '__main__':
    unittest.main()