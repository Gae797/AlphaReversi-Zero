'''
Config module for the environment
'''

import os

BOARD_SIZE = 8 #Number of side squares of the board
START_EMPTY_BOARD = False #Set to False to start with initial position (4 central pieces)

EDAX_ENGINE_PATH = os.path.join("edax engine","wEdax-x64.exe") #Path to Edax engine
EDAX_EVAL_PATH = os.path.join("edax engine","data","eval.dat") #Path to Edax evaluation weights
XOT_PATH = os.path.join("edax engine","xot.txt") #Path to XOT's positions
