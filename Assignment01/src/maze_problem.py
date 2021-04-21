'''
Specifies a MazeProblem as parameterized by a given grid maze,
assuming that an agent's legal actions can move them one tile in
any cardinal direction
'''

from constants import Constants
from maze_clause import *

class MazeProblem:

    ##################################################################
    # Class Constants
    ##################################################################

    # Static COST_MAP for maze components and the cost to move onto them
    # Any maze block not listed here is assumed to have a cost of 1
    # HINT: You can add block types to this!
    COST_MAP = {Constants.PIT_BLOCK: Constants.get_pit_penalty()}


    ##################################################################
    # Constructor
    ##################################################################

    def __init__(self, maze, explored, curious, warning, env):
        """
        Constructs a new pathfinding problem from a maze
        :maze: a list of list of strings containing maze elements
        """
        self.env = env
        self.maze = maze
        self.explored = explored
        self.curious = curious
        self.warning = warning

    ##################################################################
    # Methods
    ##################################################################

    def transitions(self, state):
        """
        Given some state s, the transitions will be represented as a list of tuples
        of the format:
        [(action1, cost_of_action1, result(action1, s)), ...]
        For example, if an agent is at state (1, 1), and can only move right and down
        into clear tiles (.), then the transitions for that s = (1, 1) would be:
        [("R", 1, (2, 1)), ("D", 1, (1, 2))]
        :state: A maze location tuple
        """
        s = state
        possible = [("U", (s[0], s[1]-1)), ("D", (s[0], s[1]+1)), ("L", (s[0]-1, s[1])), ("R", (s[0]+1, s[1]))]
        return [(s[0], self.cost(s[1]), s[1]) for s in possible if self.maze[s[1][1]][s[1][0]] != Constants.WALL_BLOCK]

    def cost(self, state):
        """
        Returns the cost of moving onto the given state, and employs
        the MazeProblem's COST_MAP
        :state: A maze location tuple

        NOTE: How did we find these values for the heuristic?
        We used heuristic_finder.py (TM @Andrew Rossell 2020)
        See that file for a few more comments.
        """
        if self.maze[state[1]][state[0]] == "P":
            return 18
        elif state in self.explored:
            return 1
        elif state in self.curious:
            if self.curious.get(state) == 1:
                return -4.5
            elif self.curious.get(state) == 2:
                return -4.9
            elif self.curious.get(state) == 3:
                return -4.9
            elif self.curious.get(state) == 4:
                return -4.9
        elif state in self.warning:
            return 10.5
        else:
            return 0.5
