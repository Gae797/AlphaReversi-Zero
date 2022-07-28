from src.environment.board import Board
import src.environment.bitboard as bitboard_handler
import src.environment.legal_moves_generator as moves_generator

import time

board = Board()
a = bitboard_handler.bitboard_to_numpy_matrix("0b1011001011010101")
print(a)

#start_time = time.time()
#print("--- %s seconds ---" % (time.time() - start_time))
