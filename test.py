from src.environment.board import Board
from src.gui.window import Window
import src.environment.bitboard as bitboard_handler
import src.environment.rules.legal_moves_generator as moves_generator
from src.game import Game
from src.agents.random_agent import RandomAgent
from src.agents.greedy_network_agent import GreedyNetworkAgent
from src.environment.config import *
from src.rl.config import *
from src.rl.training.trainer import Trainer

import time
import os

if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()

    #white_agent = RandomAgent()
    #black_agent = GreedyNetworkAgent(BOARD_SIZE, 9)
    #game = Game(white_agent, black_agent)

    #game.play()

    #start_time = time.time()
    #print("--- %s seconds ---" % (time.time() - start_time))
