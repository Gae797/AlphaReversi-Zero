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

import time
import os

if __name__ == '__main__':

    #trainer = Trainer()
    #trainer.run()

    agent_1 = AlphaReversiAgent(BOARD_SIZE, 9, None, 10)
    agent_2 = EdaxAgent(10)

    #match = Match(agent_1, agent_2, 10, use_gui=False)
    #match.play()

    game = Game(agent_1, agent_2, show_names=True, start_from_random_position=True)
    game.play_game()

    #start_time = time.time()
    #print("--- %s seconds ---" % (time.time() - start_time))
