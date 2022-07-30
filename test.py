from src.environment.board import Board
from src.environment.gui import Window
import src.environment.bitboard as bitboard_handler
import src.environment.rules.legal_moves_generator as moves_generator

import time

board = Board()
window = Window(draw_legal_moves=True)

#start_time = time.time()
window.update(board)
board = board.move(20)
#print("--- %s seconds ---" % (time.time() - start_time))
