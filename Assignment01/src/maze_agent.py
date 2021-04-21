'''
BlindBot MazeAgent meant to employ Propositional Logic,
Search, Planning, and Active Learning to navigate the
Maze Pitfall problem
'''

import time
import random
from pathfinder import *
from maze_problem import *
from queue import Queue
from queue import PriorityQueue

from maze_clause import *
from maze_knowledge_base import *

from our_own_damn_constants import OurOwnDamnConstants
class MazeAgent:

    ##################################################################
    # Constructor
    ##################################################################
    def __init__ (self, env):
        self.env  = env
        self.loc  = env.get_player_loc()
        self.goal = env.get_goal_loc()
        self.maze = env.get_agent_maze()
        self.plan = Queue()

        '''
         Lists of tile locations used for keeping track of certain distinguishable
         tile types in the maze of tiles in Maze. We use these for feeding the A* heuristic
         different costs for different tile types.
           - explored: a list of tile locations that we have complete information about
               either because we have visited this location and _perceived it
               or because we can infer its value absolutely
           - curious: a dictionary of tile locs that we know are ¬P (are safe to visit),
               but we are not sure if they are ., 1, or 2. This loc is stored as the key.
               The value of each key is a score given by the number of adjacent tiles
               that (P at (loc) is True) is False for and that are also not explored
           - warning: a list of tile locs of which we know that P is not False. These
               are either True for P or unknown for P. We don't need to differentiate
               when P is true and when P is unknown because we can differentiate them
               by querying the tile which we do in in maze_problem
           NOTE: all locs stored in curious and warning are adjacent to locs we have
                 _perceived. All tiles that are not members of any of these lists are
                 regarded as "else" in maze_problem and given their own cost.
        '''
        self.explored = list()
        self.curious = dict()
        self.warning = list()

        self.kb = MazeKnowledgeBase()

        # G, @, and X are all ¬P
        self._tile_false("P", self.loc)
        self._tile_false("P", self.goal)
        for r in range(len(self.maze)):
            for c in range(len(self.maze[r])):
                if self.maze[r][c] == "X":
                    self._tile_false("P", (c,r))


    ##################################################################
    # Private Helper Methods
    ##################################################################

    ###################
    # General Helpers #
    ###################
    def _tile_true(self, tile, loc):
        '''
        This helper method does all the necessary procedures needed when setting
        a tile at a location to True in the KB.
        '''
        # checks if the loc is valid
        if self._loc_in_maze(loc):
            self.kb.tell(MazeClause([((tile, loc), True)]))
            if tile == ".":
                # for "." all other possible tiles (P, 1, 2) must be false
                for const in OurOwnDamnConstants.RUN_MAZE_CONST:
                    self.kb.tell(MazeClause([((const, loc), const == tile)]))
            else:
            # for all locs where tile != ".", "." is False
                self.kb.tell(MazeClause([((".", loc), False)]))
                if tile == "P" and loc in self.curious:
                    del self.curious[loc]
            self._update_tile(tile, loc)

    def _tile_false(self, tile, loc):
        '''
        This helper method fulfills all the necessary procedures needed when setting a
        a tile to False at a given location in KB.
        '''
        # checks if the loc is valid
        if self._loc_in_maze(loc):
            self.kb.tell(MazeClause([((tile, loc), False)]))
            if (tile == "P"):
                if loc in self.warning:
                    self.warning.remove(loc)
                if loc not in self.curious and loc not in self.explored:
                    self._new_curious(loc)
            accounted = self.kb.false_at_loc(loc)
            if len(accounted) == len(OurOwnDamnConstants.RUN_MAZE_CONST) - 1:
                for tile_string in OurOwnDamnConstants.RUN_MAZE_CONST:
                    if tile_string not in accounted:
                        self._update_exploration(loc)
                        self._tile_true(tile_string, loc)

    def _update_tile(self, tile, loc):
        '''
        This helper method updates a tile at a location in the maze appropriately
        This method is called after we gather new information about a location with certainty enough to
        update the tile value (KB at given location would have at least one thing true).
        '''
        # Does not modify X, G, and @ tiles even though we can have knowledge at those locations
        if (self.maze[loc[1]][loc[0]] != "X" and self.maze[loc[1]][loc[0]] != "G" and self.maze[loc[1]][loc[0]] != "@"):
            # P and . are always displayed as themselves
            if tile == "P" or tile == ".":
                self.maze[loc[1]][loc[0]] = tile
            elif (self.kb.within(MazeClause([(("P", loc), False)]))):
                # If 2 is True, and 1 is True at a tile, it is displayed as 1
                if (tile == 2 and self.maze[loc[1]][loc[0]] != "1") or tile == "1":
                    self.maze[loc[1]][loc[0]] = tile

    def _get_adjacent(self, loc):
        '''
        Returns a list of locs of the tiles adjacent to the input loc.
        '''
        return [(loc[0]+1, loc[1]), (loc[0]-1, loc[1]), (loc[0], loc[1]+1), (loc[0], loc[1]-1)]

    def _loc_in_maze(self, loc):
        '''
        Returns True if the given loc is within the maze, false otherwise
        '''
        return loc[1] >= 0 and loc[1] < len(self.maze) and loc[0] >= 0 and loc[0] < len(self.maze[0])

    ########################
    # Helpers for KB Rules #
    ########################
    def _rule_adjacent(self, loc, tile, tile_bool):
        '''
        Rule helper method that just updates adjacent tiles to the loc
        with the tile and tile_bool combo
        '''
        if tile_bool:
            for adj_loc in self._get_adjacent(loc):
                self._tile_true(tile, adj_loc)
        else:
            for adj_loc in self._get_adjacent(loc):
                self._tile_false(tile, adj_loc)

    def _rule_three(self, loc, tile_type, tile_bool, res_type, res_bool):
        '''
        This is a rule helper method that when 3/4 tiles around a loc are
        some given tile_type, tile_bool, it updates the remaining 1 tile to
        res_type and res_bool
        '''
        count = 0
        res = list()
        for adj_loc in self._get_adjacent(loc):
            if self.kb.within(MazeClause([((tile_type, adj_loc), tile_bool)])):
                count += 1
            else:
                res.append(adj_loc)
        if count == 3 and len(res) == 1:
            if res_bool:
                self._tile_true(res_type, res[0])
            else:
                self._tile_false(res_type, res[0])

    def _rule_across(self, loc, before_tile, before_bool, after_tile, after_bool):
        '''
        Rule helper method that is only used for the case of . 1 ?
        which can be updated to . 1 ¬P using this method.
        Uses before_tile and before_bool to determine if after_tile,
        after_bool should be added to the KB for the tile across from
        the before_loc
        '''
        adj_list = self._get_adjacent(loc)
        across_res_list = [(loc[0]-1, loc[1]), (loc[0]+1, loc[1]), (loc[0], loc[1]-1), (loc[0], loc[1]+1)]
        for i in range(len(adj_list)):
            if self.kb.within(MazeClause([((before_tile, adj_list[i]), before_bool)])):
                if after_bool:
                    self._tile_true(after_tile, across_res_list[i])
                else:
                    self._tile_false(after_tile, across_res_list[i])

    def _rule_three_one(self, loc, three_tile, three_bool, one_tile, one_bool, new_tile, new_bool):
        '''
        Rule helper method that takes 4 known adjacent tiles to a loc
        and if three of them match the three_tile, three_bool combo
        and one matches the one_tile, one_bool combo, it updates the tile
        across from the one_tile with the new_tile, new_bool
        (better understood looking at the diagram in the _update_kb_maze()
        method).
        '''
        three = list()
        one = list()
        adj_list = self._get_adjacent(loc)
        _rule_two_away_loc = [(loc[0]+2, loc[1]), (loc[0]-2, loc[1]), (loc[0], loc[1]+2), (loc[0], loc[1]-2)]
        for i in range(len(adj_list)):
            if self.kb.within(MazeClause([((three_tile, adj_list[i]), three_bool)])):
                three.append(i)
            if self.kb.within(MazeClause([((one_tile, adj_list[i]), one_bool)])):
                one.append(i)
        if len(three) == 3 and len(one) == 1:
            new_loc  = _rule_two_away_loc[one[0]]
            if new_bool:
                self._tile_true(new_tile, new_loc)
            else:
                self._tile_false(new_tile, new_loc)

    def _rule_adj_x_not_1(self, loc):
        '''
        Rule helper method that only applies to 2s at locations
        adjacent to walls ("X"). For any 2 adjacent to a wall,
        the wall is ¬1. Walls can possibly be 1 if there is an
        adjacent pit, but not in this case.

        '''
        for adj_loc in self._get_adjacent(loc):
            if self.maze[adj_loc[1]][adj_loc[0]] == "X":
                self._tile_false("1", adj_loc)

    def _rule_two_away(self, loc, new_tile, new_bool):
        '''
        Rule helper method for when we want to update the KB
        about tiles that are two locs away (cardinal directions)
        '''
        if new_bool:
            self._tile_true(new_tile, (loc[0]-2,loc[1]))
            self._tile_true(new_tile, (loc[0]+2,loc[1]))
            self._tile_true(new_tile, (loc[0],loc[1]-2))
            self._tile_true(new_tile, (loc[0],loc[1]+2))
        else:
            self._tile_false(new_tile, (loc[0]-2,loc[1]))
            self._tile_false(new_tile, (loc[0]+2,loc[1]))
            self._tile_false(new_tile, (loc[0],loc[1]-2))
            self._tile_false(new_tile, (loc[0],loc[1]+2))

    def _perceive(self, tile, loc):
        '''
        Helper method that defines what the processes
        that happen to the KB/maze when we perceive a loc
        '''
        self._tile_true(tile, loc)
        if tile != "P":
            self._tile_false("P", loc)
            if tile == "1":
                self._tile_false(".", loc)
            elif tile == "2":
                self._tile_false("1", loc)
                self._tile_false(".", loc)
            elif tile == ".":
                self._tile_false("1", loc)
                self._tile_false("2", loc)

    #######################
    # KB and Maze Updater #
    #######################

    def _update_kb_maze(self, perception):
        '''
        Updates the entire knowledge base and agent maze with respect to the agent's perception.
        This method serves as an inference engine that uses different rules (which are described
        in comments below) to deduce new information that is added to the KB.

        NOTE 1: It should be noted that while tiles can only display one tile type at a time, a
        certain tile location can be true of several different tile types at once. For example,
        if there are two pits adjacent to one another they are both P and also both 1 since they
        are 1 away from a pit as well. This is important to the inferences we can make.

        NOTE 2: We realize now that there is probably a much more computationally efficient way
        of doing this (i.e. so that we don't have to loop through the entire maze at each step)
        like only updating tiles that are adjacent to tiles that have been updated. But we realized
        this too late and what is here, works so it is what it is :/
        '''
        for r in range(len(self.maze)):
            for c in range(len(self.maze[r])):
                if r > 0 and r < len(self.maze) - 1 and c > 0 and c < len(self.maze[0]) - 1:
                    # Add our new perception to the kb and maze
                    if perception.get("loc") == (c,r):
                        self._perceive(perception.get("tile"), (c,r))

                    if self.kb.within(MazeClause([(("1", (c,r)), False)])):
                        #    ?              ¬P
                        # ? ¬1 ?   ==>   ¬P ¬1 ¬P
                        #    ?              ¬P
                        self._rule_adjacent((c,r), "P", False)

                    if self.kb.within(MazeClause([((".", (c,r)), True)])):
                        #      ?                  ¬P
                        #      ?                   ?
                        # ? ?  .  ? ?   ==>   ¬P ? . ? ¬P
                        #      ?                   ?
                        #      ?                  ¬P
                        self._rule_two_away((c,r), "P", False)

                    if self.kb.within(MazeClause([(("2", (c,r)), True)])):
                        #    ?              ?
                        # X  2 ?   ==>   ¬1 2 ?
                        #    X             ¬1
                        self._rule_adj_x_not_1((c,r))

                        #    .             .
                        # .  2 ?   ==>   . 2 1
                        #    .             .
                        self._rule_three((c,r), ".", True, "1", True)

                        #      ?                   ?
                        #     ¬1                  ¬1
                        # ? ¬1 2  1 ?   ==>   ? ¬1 2 1 P
                        #     ¬1                  ¬1
                        #      ?                   ?
                        self._rule_three_one((c,r), "1", False, "1", True, "P", True)

                    if self.kb.within(MazeClause([(("P", (c,r)), True)])):
                        #      ?                  2
                        #      ?                  1
                        # ? ?  P  ? ?   ==>   2 1 P 1 2
                        #      ?                  1
                        #      ?                  2
                        self._rule_adjacent((c,r), "1", True)
                        self._rule_two_away((c,r), "2", True)

                    if self.kb.within(MazeClause([(("1", (c,r)), True)])):
                        # 1  .  ?   ==>   1 . ¬P
                        self._rule_across((c,r), ".", True, "P", False)

                        #   ¬P             ¬P
                        # ¬P 1 ?   ==>   ¬P 1 P
                        #   ¬P             ¬P
                        self._rule_three((c,r), "P", False, "P", True)

    #######################
    # Exploration Updater #
    #######################

    def _new_curious(self, loc):
        '''
        Helper method for adding something to the curious dict
        Each member of the curious dict is accompanied with a score
        that is equal to the number of adjacent squares that are
        both unexplored and are Pit at loc is not True
        '''
        num_unexplored = 0
        for adj_loc in self._get_adjacent(loc):
            if self._loc_in_maze(adj_loc) and self.maze[adj_loc[1]][adj_loc[0]] != "X" and adj_loc not in self.explored and not self.kb.within(MazeClause([(("P", adj_loc), True)])):
                num_unexplored += 1
        self.curious[loc] = num_unexplored


    def _update_exploration(self, loc):
        '''
        Helper method for updating our different data structures
        for when we explore (or infer an exploration) of a tile.
        '''
        if loc not in self.explored:
            self.explored.append(loc)
        if loc in self.curious:
            del self.curious[loc]
        if loc in self.warning:
            self.warning.remove(loc)

        for adj_loc in self._get_adjacent(loc):
            if self._loc_in_maze(adj_loc) and self.maze[adj_loc[1]][adj_loc[0]] != "X" and adj_loc not in self.explored and adj_loc not in self.curious and adj_loc not in self.warning:
                if self.kb.within(MazeClause([(("P", (adj_loc[0], adj_loc[1])), False)])):
                    self._new_curious(adj_loc)
                else:
                    self.warning.append(adj_loc)

    ##################################################################
    # Main Methods
    ##################################################################

    def get_next_move(self):
        """
        Returns the next move in the plan, if there is one, otherwise None
        [!] You should NOT need to modify this method -- contact Dr. Forney
            if you're thinking about it
        """
        return None if self.plan.empty() else self.plan.get()

    def think(self, perception):
        """
        think is parameterized by the agent's perception of the tile type
        on which it is now standing, and is called during the environment's
        action loop. This method is the chief workhorse of your MazeAgent
        such that it must then generate a plan of action from its current
        knowledge about the environment.

        :perception: A dictionary providing the agent's current location
        and current tile type being stood upon, of the format:
          {"loc": (x, y), "tile": tile_type}
        """

        # update player location
        self.loc = self.env.get_player_loc()

        # update KB with perception
        self._update_kb_maze(perception)

        # update explore at self.loc and make suitable adjustments to the
        # other data structures are made as well
        self._update_exploration(self.loc)

        # adds the first step in the A* to the path! (the KB updates at
        # each step and we call A* each step, so we only care about A*s
        # first move)
        if self.loc != self.goal:
            mp = MazeProblem(self.maze, self.explored, self.curious, self.warning, self.env)
            goal_path = pathfind(mp, self.loc, self.goal)
            self.plan.put(goal_path[1][0])
