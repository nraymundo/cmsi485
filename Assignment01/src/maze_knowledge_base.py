'''
maze_knowledge_base.py

Specifies a simple, Conjunctive Normal Form Propositional
Logic Knowledge Base for use in Grid Maze pathfinding problems
with side-information.
'''
from maze_clause import MazeClause
import unittest
import copy

class MazeKnowledgeBase:

    def __init__ (self):
        self.clauses = set()

    def tell (self, clause):
        """
        Adds the given clause to the CNF MazeKnowledgeBase
        Note: we expect that no clause added this way will ever
        make the KB inconsistent (you need not check for this)
        """
        return self.clauses.add(clause)

    def false_at_loc(self, loc):
        s = set()
        """
        return all clauses at location that are false
        """
        for c in self.clauses:
            if(c.props.values == False):
                s.add(c.props.keys()[0])
                
        return s

    def within(self, query):
        """
        check if the given query is in the KB
        """
        for c in self.clauses:
            if query == c:
                return True
        return False

    def ask (self, query):
        """
        Given a MazeClause query, returns True if the KB entails
        the query, False otherwise
        """
        not_query = copy.deepcopy(query)
        for key in not_query.props:
            not_query.props[key] = not query.props[key]
        clauses = copy.deepcopy(self.clauses)
        clauses.add(not_query)
        new = set()
        while True:
            for i in clauses:
                for j in clauses:
                    res = MazeClause.resolve(i, j)
                    if MazeClause([]) in res:
                        return True
                    new = new.union(res)
            if new.issubset(clauses):
                return False
            clauses = clauses.union(new)

class MazeKnowledgeBaseTests(unittest.TestCase):
    def test_mazekb1(self):
        kb = MazeKnowledgeBase()
        kb.tell(MazeClause([(("X", (1, 1)), True)]))
        self.assertTrue(kb.ask(MazeClause([(("X", (1, 1)), True)])))

    def test_mazekb2(self):
        kb = MazeKnowledgeBase()
        kb.tell(MazeClause([(("X", (1, 1)), False)]))
        kb.tell(MazeClause([(("X", (1, 1)), True), (("Y", (1, 1)), True)]))
        self.assertTrue(kb.ask(MazeClause([(("Y", (1, 1)), True)])))

    def test_mazekb3(self):
        kb = MazeKnowledgeBase()
        kb.tell(MazeClause([(("X", (1, 1)), False), (("Y", (1, 1)), True)]))
        kb.tell(MazeClause([(("Y", (1, 1)), False), (("Z", (1, 1)), True)]))
        kb.tell(MazeClause([(("W", (1, 1)), True), (("Z", (1, 1)), False)]))
        kb.tell(MazeClause([(("X", (1, 1)), True)]))
        self.assertTrue(kb.ask(MazeClause([(("W", (1, 1)), True)])))
        self.assertFalse(kb.ask(MazeClause([(("Y", (1, 1)), False)])))

    def test_mazekb4(self):
        kb = MazeKnowledgeBase()
        kb.tell(MazeClause([(("X", (1, 1)), True)]))
        kb.tell(MazeClause([(("X", (1, 1)), True)]))
        self.assertTrue(len(kb.clauses) == 1)


if __name__ == "__main__":
    unittest.main()