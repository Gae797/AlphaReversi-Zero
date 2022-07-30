import numpy as np

from src.environment.config import *
import src.environment.bitboard as bitboard_handler
import src.environment.rules.legal_moves_generator as moves_generator
import src.environment.rules.reverter as reverter

class Board:

    def __init__(self, white_pieces=None, black_pieces=None, turn=None):

        self.white_pieces = white_pieces
        self.black_pieces = black_pieces
        self.turn = turn

        if self.white_pieces is None or self.black_pieces is None or self.turn is None:
            self.reset()

        self.empty_squares = self.generate_empty_squares()

        self.legal_moves = self.generate_legal_moves()
        self.is_terminal = len(self.legal_moves["array"])==0

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

        bitboard_legal_moves = moves_generator.complete_search(mover_pieces, opponent_pieces, self.empty_squares)

        array_legal_moves = bitboard_handler.bitboard_to_numpy_array(bitboard_legal_moves)
        matrix_legal_moves = bitboard_handler.bitboard_to_numpy_matrix(bitboard_legal_moves)
        indices_legal_moves = np.where(array_legal_moves==1)[0].tolist()

        legal_moves = {
        "bitboard": bitboard_legal_moves,
        "array": array_legal_moves,
        "matrix": matrix_legal_moves,
        "indices": indices_legal_moves
        }

        return legal_moves

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

    def get_state(self, legal_moves_format="array"):

        if not legal_moves_format in ["array", "matrix", "bitboard", "indices"]:
            raise "Invalid legal moves format: {}".format(legal_moves_format)

        white_pieces = bitboard_handler.bitboard_to_numpy_matrix(self.white_pieces)
        black_pieces = bitboard_handler.bitboard_to_numpy_matrix(self.black_pieces)
        turn = np.ones((BOARD_SIZE, BOARD_SIZE)) if self.turn else np.zeros((BOARD_SIZE, BOARD_SIZE))
        legal_moves = self.legal_moves[legal_moves_format]

        return white_pieces, black_pieces, turn, legal_moves, self.reward

    def move(self, move_number):

        if move_number not in self.legal_moves["indices"]:
            raise "Invalid move!"

        move_bitboard = "0b" + "0"*move_number + "1" + "0"*(BOARD_SIZE*BOARD_SIZE - move_number - 1)

        #Add piece
        mover_pieces = self.white_pieces if self.turn else self.black_pieces
        opponent_pieces = self.black_pieces if self.turn else self.white_pieces
        mover_pieces = bitboard_handler.bitwise([mover_pieces, move_bitboard], "or")

        #Revert pieces in between
        reverted_pieces = reverter.complete_search(mover_pieces, opponent_pieces, move_bitboard)
        mover_pieces = bitboard_handler.bitwise([mover_pieces, reverted_pieces], "or", True)
        opponent_pieces = bitboard_handler.bitwise([opponent_pieces, reverted_pieces], "xor", True)

        #Create and return new state
        if self.turn:
            white_pieces = mover_pieces
            black_pieces = opponent_pieces
            turn = False
        else:
            white_pieces = opponent_pieces
            black_pieces = mover_pieces
            turn = True

        new_board = Board(white_pieces, black_pieces, turn)

        return new_board
