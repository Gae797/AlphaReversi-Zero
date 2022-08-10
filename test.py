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
from src.rl.training.local_trainer import Trainer
from src.evaluation.match import Match
from src.evaluation.evaluator import Evaluator, evaluate_generations

import time
import os

if __name__ == '__main__':

    trainer = Trainer()
    trainer.run()

    #generation_1 = 2
    #generation_2 = 4

    #weights_1 = os.path.join(WEIGHTS_PATH, "Generation {}".format(generation_1), "variables")
    #weights_2 = os.path.join(WEIGHTS_PATH, "Generation {}".format(generation_2), "variables")

    #agent_1 = AlphaReversiAgent(BOARD_SIZE, N_RESIDUAL_BLOCKS, weights_1, 200, name="Generation {}".format(generation_1))
    #agent_2 = AlphaReversiAgent(BOARD_SIZE, N_RESIDUAL_BLOCKS, weights_2, 200, name="Generation {}".format(generation_2))

    #match = Match(agent_1, agent_2, 10, use_gui=True, start_from_random_position=False)
    #match.play()

    #evaluate_generations([1,2,3],10,10,10,True)

    #start_time = time.time()
    #print("--- %s seconds ---" % (time.time() - start_time))
