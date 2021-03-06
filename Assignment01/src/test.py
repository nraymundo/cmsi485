'''
maze_clause.py

Specifies a Propositional Logic Clause formatted specifically
for Grid Maze Pathfinding problems. Clauses are a disjunction of
MazePropositions (2-tuples of (symbol, location)) mapped to
their negated status in the sentence.
'''
import unittest

class MazeClause:

    def __init__(self, props):
        """
        Constructor parameterized by the propositions within this clause;
        argument props is a list of MazePropositions, like:
        [(("X", (1, 1)), True), (("X", (2, 1)), True), (("Y", (1, 2)), False)]

        :props: a list of tuples formatted as: (MazeProposition, NegatedBoolean)
        """
        self.props = props
        self.valid = False
        for i in self.props:
            print(i)
        for tuple in (self.props):
            print(tuple[0])

    def is_valid(self):
        """
        Returns:
          - True if this clause is logically equivalent with True
          - False otherwise
        """
        return False

    def is_empty(self):
        """
        Returns:
          - True if this is the Empty Clause
          - False otherwise
        (NB: valid clauses are not empty)
        """
        # TODO: This is currently implemented incorrectly; see
        # spec for details!
        return False

    def get_prop(self, prop):
        """
        Returns:
          - None if the requested prop is not in the clause
          - True if the requested prop is positive in the clause
          - False if the requested prop is negated in the clause

        :prop: A MazeProposition as a 2-tuple formatted as: (Symbol, Location),
        for example, ("P", (1, 1))
        """

        # TODO: This is currently implemented incorrectly; see
        # spec for details!
        return False

    def __eq__(self, other):
        """
        Defines equality comparator between MazeClauses: only if they
        have the same props (in any order) or are both valid
        """
        return self.props == other.props and self.valid == other.valid

    def __hash__(self):
        """
        Provides a hash for a MazeClause to enable set membership
        """
        # Hashes an immutable set of the stored props for ease of
        # lookup in a set
        return hash(frozenset(self.props.items()))

    # Hint: Specify a __str__ method for ease of debugging (this
    # will allow you to "print" a MazeClause directly to inspect
    # its composite literals)
    # def __str__ (self):
    #     return ""

    @staticmethod
    def resolve(c1, c2):
        """
        Returns a set of MazeClauses that are the result of resolving
        two input clauses c1, c2 (Hint: result will only ever be a set
        of 0 or 1 MazeClause, but it being a set is convenient for the
        inference engine) (Hint2: returning an empty set of clauses
        is different than returning a set containing the empty clause /
        contradiction)

        :c1: A MazeClause to resolve with c2
        :c2: A MazeClause to resolve with c1
        """
        results = set()
        # TODO: This is currently implemented incorrectly; see
        # spec for details!
        return results

class MazeClauseTests(unittest.TestCase):
    def test_mazeprops1(self):
        mc = MazeClause([(("X", (1, 1)), True), (("X", (2, 1)), True), (("Y", (1, 2)), False)])
        print(mc.valid)
        self.assertFalse(mc.is_valid())

        #self.assertTrue(mc.get_prop(("X", (2, 1))))
        #self.assertFalse(mc.get_prop(("Y", (1, 2))))
        #self.assertTrue(mc.get_prop(("X", (2, 2))) is None)
        #self.assertFalse(mc.is_empty())

if __name__ == "__main__":
    unittest.main()