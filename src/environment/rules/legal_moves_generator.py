import src.environment.bitboard as bitboard_handler

from src.environment.config import *
from src.environment.rules.masks_generator import MasksGenerator

def complete_search(mover_pieces, opponent_pieces, empty_squares):

    vertical_legal_moves = vertical_search(mover_pieces, opponent_pieces, empty_squares)
    horizontal_legal_moves = horizontal_search(mover_pieces, opponent_pieces, empty_squares)
    diagonal_legal_moves = diagonal_search(mover_pieces, opponent_pieces, empty_squares)

    legal_moves = bitboard_handler.bitwise([vertical_legal_moves,horizontal_legal_moves, diagonal_legal_moves], "or", True)

    return legal_moves

def general_search(mover_pieces, opponent_pieces, empty_squares, step):

    legal_moves = bitboard_handler.empty_bitboard()

    shifted_pieces = bitboard_handler.shift(mover_pieces, step)
    valid_moves = bitboard_handler.bitwise([shifted_pieces, opponent_pieces], "and")

    for i in range(BOARD_SIZE-2):
        shifted_pieces = bitboard_handler.shift(valid_moves, step)

        found_legal_moves = bitboard_handler.bitwise([shifted_pieces, empty_squares], "and")
        legal_moves = bitboard_handler.bitwise([found_legal_moves, legal_moves], "or")

        valid_moves = bitboard_handler.bitwise([shifted_pieces, opponent_pieces], "and")

    return legal_moves

def general_diagonal_search(mover_pieces, opponent_pieces, empty_squares, diagonal_mask, up, right):

    vertical_step = BOARD_SIZE if up else -BOARD_SIZE
    horizontal_step = 1 if right else -1

    step = vertical_step + horizontal_step

    masked_mover_pieces = bitboard_handler.bitwise([mover_pieces, diagonal_mask], "and")
    masked_opponent_pieces = bitboard_handler.bitwise([opponent_pieces, diagonal_mask], "and")
    masked_empty_squares = bitboard_handler.bitwise([empty_squares, diagonal_mask], "and")

    return general_search(masked_mover_pieces, masked_opponent_pieces, masked_empty_squares, step)

def vertical_search(mover_pieces, opponent_pieces, empty_squares):

    up_legal_moves = general_search(mover_pieces, opponent_pieces, empty_squares, BOARD_SIZE)
    down_legal_moves = general_search(mover_pieces, opponent_pieces, empty_squares, -BOARD_SIZE)

    vertical_legal_moves = bitboard_handler.bitwise([up_legal_moves, down_legal_moves], "or")

    return vertical_legal_moves

def horizontal_search(mover_pieces, opponent_pieces, empty_squares):

    rotated_mover_pieces = bitboard_handler.rotate_90(mover_pieces)
    rotated_opponent_pieces = bitboard_handler.rotate_90(opponent_pieces)
    rotated_empty_squares = bitboard_handler.rotate_90(empty_squares)

    right_legal_moves = general_search(rotated_mover_pieces, rotated_opponent_pieces, rotated_empty_squares, BOARD_SIZE)
    left_legal_moves = general_search(rotated_mover_pieces, rotated_opponent_pieces, rotated_empty_squares, -BOARD_SIZE)

    rotated_horizontal_legal_moves = bitboard_handler.bitwise([right_legal_moves, left_legal_moves], "or")

    horizontal_legal_moves = bitboard_handler.rotate_270(rotated_horizontal_legal_moves)

    return horizontal_legal_moves

def diagonal_search(mover_pieces, opponent_pieces, empty_squares):

    legal_moves = []

    masks_bottom_up, masks_top_down = MasksGenerator.get_masks()

    for mask in masks_bottom_up:
        legal_moves.append(general_diagonal_search(mover_pieces, opponent_pieces, empty_squares, mask, True, True))
        legal_moves.append(general_diagonal_search(mover_pieces, opponent_pieces, empty_squares, mask, False, False))

    for mask in masks_top_down:
        legal_moves.append(general_diagonal_search(mover_pieces, opponent_pieces, empty_squares, mask, True, False))
        legal_moves.append(general_diagonal_search(mover_pieces, opponent_pieces, empty_squares, mask, False, True))

    return bitboard_handler.bitwise(legal_moves, "or")
