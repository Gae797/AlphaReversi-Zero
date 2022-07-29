from src.environment.board import Board
import src.environment.bitboard as bitboard_handler
import src.environment.legal_moves_generator as moves_generator

import time

board = Board()
start_time = time.time()
board = board.move(20)
print("--- %s seconds ---" % (time.time() - start_time))
