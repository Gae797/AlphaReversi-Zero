import numpy as np

from src.environment.config import *
import src.environment.bitboard as bitboard_handler
import src.environment.legal_moves_generator as moves_generator

class Board:

    def __init__(self, white_pieces=None, black_pieces=None, turn=None):

        self.white_pieces = white_pieces
        self.black_pieces = black_pieces
        self.turn = turn

        if self.white_pieces is None or self.black_pieces is None or self.turn is None:
            self.reset()

        self.empty_squares = self.generate_empty_squares()

        self.legal_moves = self.generate_legal_moves()
        self.is_terminal = len(self.legal_moves)==0

        self.reward = self.generate_reward()

    def reset(self):

        if START_EMPTY_BOARD:
            self.white_pieces = bitboard_handler.empty_bitboard()
            self.black_pieces = bitboard_handler.empty_bitboard()

        else:
            self.white_pieces, self.black_pieces = bitboard_handler.starting_bitboard()

        self.turn = False #Black to move

    def generate_empty_squares(self):

        occupied_squares = bitboard_handler.bitwise([self.white_pieces, self.black_pieces],"or",True)

        return bitboard_handler.negate(occupied_squares)

    def generate_legal_moves(self):

        mover_pieces = self.white_pieces if self.turn else self.black_pieces
        opponent_pieces = self.black_pieces if self.turn else self.white_pieces

        vertical_legal_moves = moves_generator.vertical_search(mover_pieces, opponent_pieces, self.empty_squares)
        horizontal_legal_moves = moves_generator.horizontal_search(mover_pieces, opponent_pieces, self.empty_squares)
        diagonal_legal_moves = moves_generator.diagonal_search(mover_pieces, opponent_pieces, self.empty_squares)

        legal_moves = bitboard_handler.bitwise([vertical_legal_moves,horizontal_legal_moves, diagonal_legal_moves], "or", True)

        return bitboard_handler.bitboard_to_numpy_array(legal_moves)

    def generate_reward(self):

        if not self.is_terminal:
            return None

        else:
            white_count = bitboard_handler.count_ones(self.white_pieces)
            black_count = bitboard_handler.count_ones(self.black_pieces)

            if white_count==black_count:
                return 0

            elif white_count>black_count:
                return 1

            else:
                return -1

    def get_state():

        white_pieces = bitboard_handler.bitboard_to_numpy_matrix(self.white_pieces)
        black_pieces = bitboard_handler.bitboard_to_numpy_matrix(self.black_pieces)
        turn = np.ones((BOARD_SIZE, BOARD_SIZE)) if self.turn else np.zeros((BOARD_SIZE, BOARD_SIZE))

        return white_pieces, black_pieces, turn, self.legal_moves, self.reward

    def move(move_number):

        if move_number not in self.legal_moves.tolist():
            raise "Invalid move!"

        #TODO: add piece
        #TODO: revert pieces in between
        #TODO: create and return new state
