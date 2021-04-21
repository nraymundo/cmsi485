'''
NOTE: This file was originally a modification on environment.py so most of the code
for environment.py is copied here because we couldn't be bothered to actually
make a unique seperate finder.

Heuristic Finder iterates through ranges of possible heuristic values,
and solves all the mazes with these cost values, returning the heuristic
that gives the best total score.
The cost values we found (see maze_problem.py) may seem a bit counterintuitive.
But don't ask us why! The computer found these values.

NOTE: I spend in total probably 60+ hours running this brilliantly computationally
efficient program. :) -Damián
'''




from maze_agent import MazeAgent
import queue
import numpy as np
import time
import os
import re
import sys
import time
import copy
from constants import Constants
class HeuristicEnvironment:

    ##################################################################
    # Constructor
    ##################################################################

    def __init__(self, maze, tick_length=0, verbose=False, pitCost=19, exploreCost=1, curiousCost1=-5, curiousCost2=-4.5, curiousCost3=-6, curiousCost4=-6, warningCost=10, otherCost=1):
        """
        Initializes the environment from a given maze, specified as an
        array of strings with maze elements
        :maze: The array of strings specifying the challenge
        :tick_length: The duration between agent decisions, in seconds
        :verbose: Whether or not the maze updates will be printed
        """
        self._maze = maze
        self._rows = len(maze)
        self._cols = len(maze[0])
        self._tick_length = tick_length
        self._verbose = verbose
        self._pits = set()
        self._goals = set()
        self._walls = set()
        self.pitCost = pitCost
        self.exploreCost = exploreCost
        self.curiousCost1 = curiousCost1
        self.curiousCost2 = curiousCost2
        self.curiousCost3 = curiousCost3
        self.curiousCost4 = curiousCost4
        self.warningCost = warningCost
        self.otherCost = otherCost

        # Scan for pits and goals in the input maze
        for (row_num, row) in enumerate(maze):
            for (col_num, cell) in enumerate(row):
                if cell == Constants.WALL_BLOCK:
                    self._walls.add((col_num, row_num))
                if cell == Constants.GOAL_BLOCK:
                    self._goals.add((col_num, row_num))
                if cell == Constants.PIT_BLOCK:
                    self._pits.add((col_num, row_num))
                if cell == Constants.PLR_BLOCK:
                    self._player_loc = self._initial_loc = (col_num, row_num)
        self._spcl = self._pits | self._goals | self._walls
        self._wrn1_tiles = self._get_wrn_set(
            [self._get_adjacent(loc, 1) for loc in self._pits])
        self._wrn2_tiles = self._get_wrn_set(
            [self._get_adjacent(loc, 2) for loc in self._pits])

        # Initialize the MazeAgent and ready simulation!
        self._goal_reached = False
        self._ag_maze = self._make_agent_maze()
        self._ag_tile = Constants.SAFE_BLOCK
        # Easier to change elements in this format
        self._maze = [list(row) for row in maze]
        self._og_maze = copy.deepcopy(self._maze)
        self._og_maze[self._player_loc[1]
                      ][self._player_loc[0]] = Constants.SAFE_BLOCK
        for (c, r) in self._wrn2_tiles:
            self._og_maze[r][c] = Constants.WRN_BLOCK_2
        for (c, r) in self._wrn1_tiles:
            self._og_maze[r][c] = Constants.WRN_BLOCK_1
        self._agent = MazeAgent(self)

    ##################################################################
    # Methods
    ##################################################################

    def get_player_loc(self):
        """
        Returns the player's current location as a maze tuple
        """
        return self._player_loc

    def get_goal_loc(self):
        return next(iter(self._goals))

    def get_agent_maze(self):
        """
        Returns the agent's mental model of the maze, without key
        components revealed that have yet to be explored. Unknown
        spaces are filled with "?"
        """
        return self._ag_maze

    def start_mission(self):
        """
        Manages the agent's action loop and the environment's record-keeping
        mechanics
        """
        score = 0
        while (score > Constants.get_min_score()):
            time.sleep(self._tick_length)

            # Get player's next move in their plan, then execute
            next_act = self._agent.get_next_move()
            self._move_request(next_act)

            # Return a perception for the agent to think about and plan next
            perception = {"loc": self._player_loc, "tile": self._ag_tile}
            self._agent.think(perception)

            # Assess the post-move penalty and whether or not the game is complete
            penalty = Constants.get_pit_penalty() if self._pit_test(
                self._player_loc) else Constants.get_mov_penalty()
            score = score - penalty
            if self._verbose:
                print("\nCurrent Loc: " + str(self._player_loc) +
                      " [" + self._ag_tile + "]\nLast Move: " + str(next_act) + "\nScore: " + str(score) + "\n")
            if self._goal_test(self._player_loc):
                break

        if self._verbose:
            print("[!] Game Complete! Final Score: " + str(score))
        return score

    ##################################################################
    # "Private" Helper Methods
    ##################################################################

    def _get_adjacent(self, loc, offset):
        """
        Returns a set of the 4 adjacent cells to the given loc
        """
        (x, y) = loc
        pos_locs = [(x+offset, y), (x-offset, y), (x, y+offset), (x, y-offset)]
        return list(filter(lambda loc: loc[0] >= 0 and loc[1] >= 0 and loc[0] < self._cols and loc[1] < self._rows, pos_locs))

    def _get_wrn_set(self, wrn_list):
        return {item for sublist in wrn_list for item in sublist if item not in self._spcl}

    def _update_display(self, move):
        for (rowIndex, row) in enumerate(self._maze):
            print(''.join(row) + "\t" + ''.join(self._ag_maze[rowIndex]))

    def _wall_test(self, loc):
        return loc in self._walls

    def _goal_test(self, loc):
        return loc in self._goals

    def _pit_test(self, loc):
        return loc in self._pits

    def _make_agent_maze(self):
        """
        Converts the 'true' maze into one with hidden tiles (?) for the agent
        to update as it learns
        """
        sub_regexp = "[" + Constants.PIT_BLOCK + Constants.SAFE_BLOCK + "]"
        return [list(re.sub(sub_regexp, Constants.UNK_BLOCK, r)) for r in self._maze]

    def _move_request(self, move):
        old_loc = self._player_loc
        new_loc = old_loc if move == None else tuple(
            sum(x) for x in zip(self._player_loc, Constants.MOVE_DIRS[move]))
        if self._wall_test(new_loc):
            new_loc = old_loc
        self._update_mazes(self._player_loc, new_loc)
        self._player_loc = new_loc
        if self._verbose:
            self._update_display(move)

    def _update_mazes(self, old_loc, new_loc):
        self._maze[old_loc[1]][old_loc[0]
                               ] = self._og_maze[old_loc[1]][old_loc[0]]
        self._maze[new_loc[1]][new_loc[0]] = Constants.PLR_BLOCK
        self._ag_maze[old_loc[1]][old_loc[0]
                                  ] = self._og_maze[old_loc[1]][old_loc[0]]
        self._ag_maze[new_loc[1]][new_loc[0]] = Constants.PLR_BLOCK
        self._ag_tile = self._og_maze[new_loc[1]][new_loc[0]]


# Appears here to avoid circular dependency

if __name__ == "__main__":
    """
    Some example mazes with associated difficulties are
    listed below. The score thresholds given are for agents that actually use logic.
    Making a B-line for the goal on these mazes *may* satisfy the threshold listed here,
    but will not in general, more thorough tests.
    """
    mazes = [
        # Easy difficulty: Score > -20
        ["XXXXXX",
         "X...GX",
         "X..PPX",
         "X....X",
         "X..P.X",
         "X@...X",
         "XXXXXX"],

        # Medium difficulty: Score > -30
        ["XXXXXXXXX",
         "X..PGP..X",
         "X.......X",
         "X..P.P..X",
         "X.......X",
         "X..@....X",
         "XXXXXXXXX"],

        # Hard difficulty: Score > -35
        ["XXXXXXXXX",
         "X..PG...X",
         "X.......X",
         "X.P.P.P.X",
         "XP.....PX",
         "X...@...X",
         "XXXXXXXXX"],

        # Custom 1:
        ["XXXXXXXXXX",
         "X..PPP..GX",
         "X..PP....X",
         "X..P.....X",
         "X..P..P..X",
         "X@.P.....X",
         "X..P.....X",
         "X..P.....X",
         "X........X",
         "XXXXXXXXXX"],

        ["XXXXXXXX",
         "X..PPPGX",
         "X..PP..X",
         "X@.P...X",
         "X..P...X",
         "X......X",
         "XXXXXXXX"],

        ["XXXXXXXX",
         "X....P.X",
         "X.P..PGX",
         "X@P.P..X",
         "X.P..P.X",
         "X......X",
         "XXXXXXXX"],

        ["XXXXXXXXXX",
         "X.P...P.PX",
         "X....P...X",
         "X....P...X",
         "X@..P.P.GX",
         "XXXXXXXXXX"],

        ["XXXXXXXX",
         "XP..P..X",
         "X......X",
         "X...PP.X",
         "X..P.P.X",
         "X@....GX",
         "XXXXXXXX"],

        ["XXXXXXXXX",
         "X....@..X",
         "X.PP..P.X",
         "X..P..P.X",
         "XG.....PX",
         "XXXXXXXXX"],

        ["XXXXXXXXXX",
         "X.......GX",
         "XP.P.P.P.X",
         "X.P.P.P.PX",
         "X........X",
         "X.P.P.P.PX",
         "XP.P.P.P.X",
         "X........X",
         "X....@...X",
         "XXXXXXXXXX"],

        ["XXXXXXXXXX",
         "X..P....GX",
         "X..P.PPP.X",
         "X..P.....X",
         "X..PP...PX",
         "X@.P.....X",
         "X..P.PPP.X",
         "X..P.....X",
         "X........X",
         "XXXXXXXXXX"],

        ["XXXXXXXXXX",
         "XP......GX",
         "X.P......X",
         "X..P.P...X",
         "X.....PP.X",
         "X@.......X",
         "XXXXXXXXXX"],

        ["XXXXXXXXXX",
         "XP...P..PX",
         "X........X",
         "X....P...X",
         "X...P.P..X",
         "X@.P...PGX",
         "XXXXXXXXXX"],

        ["XXXXXXXXXX",
         "X....G...X",
         "X........X",
         "X........X",
         "X....P...X",
         "X........X",
         "X........X",
         "X........X",
         "X....@...X",
         "XXXXXXXXXX"],

        ["XXXXXXXXX",
         "X..PGP..X",
         "XPPPPPPPX",
         "X.......X",
         "X...@...X",
         "XXXXXXXXX"],
    ]

    class mazeScore:
        def __init__(self, score, values):
            self.score = score
            self.values = values

        def __lt__(self, other):
            return self.score > other.score

    bestMaze = queue.PriorityQueue()

    # defines the range of values we want to test!
    #pitRange = np.arange(19, 20, 1)
    #exploreRange = np.arange(1, 1.1, 0.1)
    #curiousRange1 = np.arange(-4.7, -4.5, 0.1)
    #curiousRange2 = np.arange(-4.9, -4.7, 0.1)
    #curiousRange3 = np.arange(-5.1, -4.9, 0.1)
    #curiousRange4 = np.arange(-5.3, -5.1, 0.1)
    #warningRange = np.arange(10.5, 11, 0.5)
    #otherRange = np.arange(0.9, 1.0, 0.1)

    # 19 : 12, inf > 12
    pitRange = [17.5, 18.5, 19.5]
    # 1 > 0.9, 1.1
    exploreRange = [1]
    # -4.5: -10, inf >
    curiousRange1 = [-4.5, -4]
    # -5.0: -5.0, -4.6 > -5.01. -4.59
    curiousRange2 = [-4.9]
    # -4.9: -4.99, -3.91 > -5.0, -5.1, -3.75, -3.9
    curiousRange3 = [-4.9]
    # -4.9: -4.9, -4.9 > -4.89, -4.91
    curiousRange4 = [-4.9]
    # 10.5: 9.1, 10.99> 11, 9.01
    warningRange = [10.25, 10.5, 10.75]
    # 0.9: 0.99, -0.89, > 1.0, -0.9
    otherRange = [0.9]

    total_tests = len(pitRange) * len(exploreRange) * len(curiousRange1) * len(curiousRange2) * \
        len(curiousRange3) * len(curiousRange4) * \
        len(warningRange) * len(otherRange)
    test_num = 0
    start_time = time.time()
    # WAIT STOP
    # THIS NEXT BIT IS NSFW! SCROLL AT YOUR OWN DISCRETION
    # !![ NSFW ]!!
    for p in pitRange:
        for e in exploreRange:
            for c1 in curiousRange1:
                for c2 in curiousRange2:
                    for c3 in curiousRange3:
                        for c4 in curiousRange4:
                            for w in warningRange:
                                for o in otherRange:
                                    # disgusting
                                    test_num += 1
                                    score = 0
                                    for maze_index in range(len(mazes)):
                                        env = HeuristicEnvironment(mazes[maze_index], pitCost=p, exploreCost=e, curiousCost1=c1, curiousCost2=c2, curiousCost3=c3, curiousCost4=c4, warningCost=w, otherCost=o)
                                        score += env.start_mission()
                                        # stops iterating through all the mazes if
                                        # minimum score isn't met for Forney's defined mazes
                                        if maze_index == 0:
                                            maze_easy = score
                                            if maze_easy <= -20:
                                                break
                                        elif maze_index == 1:
                                            maze_medium = score - maze_easy
                                            if maze_medium <= -30:
                                                break
                                        elif maze_index == 2:
                                            maze_hard = score - maze_medium - maze_easy
                                            if maze_hard <= -35:
                                                break                                           
                                        elif maze_index == len(mazes) - 1:
                                            print("Test: ", test_num, "/", total_tests, " |  Progress: ", int(1000 * (test_num / total_tests))/10, "%")
                                            print("Time Elapsed: ", int(time.time() - start_time), "s  |  Remaining: ", int((time.time() - start_time) * (total_tests - test_num) / test_num), "s\n")
                                            bestMaze.put(mazeScore(score, [p, e, c1, c2, c3, c4, w, o]))

    scoreSet = set()

    bestSet = bestMaze.get()
    bestScore = bestSet.score
    nextScore = bestScore

    while(nextScore == bestScore):
        scoreSet.add(bestSet)
        nextScore = bestScore
        bestSet = bestMaze.get()
        nextScore = bestSet.score

    # prints winning score + heuristic
    for x in scoreSet:
        print(f"Score: {x.score}, Values: {x.values}")
