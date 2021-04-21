from constants import Constants

class OurOwnDamnConstants:
    # PLR_BLOCK is an INIT block and not a RUN block because we only care about it as a starting location, because that is what is relevant to our inference
    # UNK_BLOCK is irrelevant because our KB is trying to find the real values of
    #INIT_MAZE_CONST = [Constants.WALL_BLOCK, Constants.GOAL_BLOCK, Constants.PLR_BLOCK]
    RUN_MAZE_CONST = [Constants.PIT_BLOCK, Constants.SAFE_BLOCK, Constants.WRN_BLOCK_1, Constants.WRN_BLOCK_2]
    #MAZE_CONST = INIT_MAZE_CONST + RUN_MAZE_CONST
