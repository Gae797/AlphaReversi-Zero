from src.environment.board import Board
from src.gui.window import Window
import src.environment.bitboard as bitboard_handler
import src.environment.rules.legal_moves_generator as moves_generator
from src.game import Game
from src.agents.random_agent import RandomAgent
from src.agents.greedy_network_agent import GreedyNetworkAgent
from src.agents.alpha_reversi_agent import AlphaReversiAgent
from src.agents.edax_agent import EdaxAgent
from src.environment.config import *
from src.rl.config import *
from src.rl.training.trainer import Trainer
from src.evaluation.match import Match
from src.evaluation.evaluator import Evaluator

import time
import os

if __name__ == '__main__':

    #trainer = Trainer()
    #trainer.run()

    evaluator = Evaluator(None, 10, 10, 10, use_gui=True)
    evaluations = evaluator.evaluate()
    print(evaluations)

    #start_time = time.time()
    #print("--- %s seconds ---" % (time.time() - start_time))
