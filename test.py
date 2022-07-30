from src.environment.board import Board
from src.gui.window import Window
import src.environment.bitboard as bitboard_handler
import src.environment.rules.legal_moves_generator as moves_generator
from src.game import Game
from src.agents.random_agent import RandomAgent

import time

white_agent = RandomAgent()
black_agent = RandomAgent()
game = Game(white_agent, black_agent)

game.play()

#start_time = time.time()
#print("--- %s seconds ---" % (time.time() - start_time))
