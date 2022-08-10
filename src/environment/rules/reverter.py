from src.environment.config import *
import src.environment.bitboard as bitboard_handler
from src.environment.rules.masks_generator import MasksGenerator

def complete_search(mover_pieces, opponent_pieces, selected_move):

    vertical_reverted_pieces = vertical_search(mover_pieces, opponent_pieces, selected_move)
    horizontal_reverted_pieces = horizontal_search(mover_pieces, opponent_pieces, selected_move)
    diagonal_reverted_pieces = diagonal_search(mover_pieces, opponent_pieces, selected_move)

    reverted_pieces = bitboard_handler.bitwise([vertical_reverted_pieces, horizontal_reverted_pieces, diagonal_reverted_pieces], "or")

    return reverted_pieces

def general_search(mover_pieces, opponent_pieces, selected_move, step):

    reverted_pieces = bitboard_handler.empty_bitboard()

    for i in range(BOARD_SIZE-1):
        selected_move = bitboard_handler.shift(selected_move, step)

        check_end = bitboard_handler.bitwise([selected_move,mover_pieces], "and")
        if check_end!=0:
            return reverted_pieces
        else:
            check_opponent = bitboard_handler.bitwise([selected_move,opponent_pieces], "and")
            if check_opponent!=0:
                reverted_pieces = bitboard_handler.bitwise([reverted_pieces, selected_move], "or")
            else:
                return 0

    return 0

def general_diagonal_search(mover_pieces, opponent_pieces, selected_move, diagonal_mask, up, right):

    masked_selected_move = bitboard_handler.bitwise([selected_move, diagonal_mask], "and")
    if masked_selected_move==0:
        return 0

    masked_mover_pieces = bitboard_handler.bitwise([mover_pieces, diagonal_mask], "and")
    masked_opponent_pieces = bitboard_handler.bitwise([opponent_pieces, diagonal_mask], "and")

    vertical_step = BOARD_SIZE if up else -BOARD_SIZE
    horizontal_step = 1 if right else -1

    step = vertical_step + horizontal_step

    return general_search(masked_mover_pieces, masked_opponent_pieces, masked_selected_move, step)

def vertical_search(mover_pieces, opponent_pieces, selected_move):

    up_reverted_pieces = general_search(mover_pieces, opponent_pieces, selected_move, BOARD_SIZE)
    down_reverted_pieces = general_search(mover_pieces, opponent_pieces, selected_move, -BOARD_SIZE)

    vertical_reverted_pieces = bitboard_handler.bitwise([up_reverted_pieces, down_reverted_pieces], "or")

    return vertical_reverted_pieces

def horizontal_search(mover_pieces, opponent_pieces, selected_move):

    rotated_mover_pieces = bitboard_handler.rotate_90(mover_pieces)
    rotated_opponent_pieces = bitboard_handler.rotate_90(opponent_pieces)
    rotated_selected_move = bitboard_handler.rotate_90(selected_move)

    right_reverted_pieces = general_search(rotated_mover_pieces, rotated_opponent_pieces, rotated_selected_move, BOARD_SIZE)
    left_reverted_pieces = general_search(rotated_mover_pieces, rotated_opponent_pieces, rotated_selected_move, -BOARD_SIZE)

    rotated_horizontal_reverted_pieces = bitboard_handler.bitwise([right_reverted_pieces, left_reverted_pieces], "or")

    horizontal_reverted_pieces = bitboard_handler.rotate_270(rotated_horizontal_reverted_pieces)

    return horizontal_reverted_pieces

def diagonal_search(mover_pieces, opponent_pieces, selected_move):

    reverted_pieces = []

    masks_bottom_up, masks_top_down = MasksGenerator.get_masks()

    for mask in masks_bottom_up:
        reverted_pieces.append(general_diagonal_search(mover_pieces, opponent_pieces, selected_move, mask, True, True))
        reverted_pieces.append(general_diagonal_search(mover_pieces, opponent_pieces, selected_move, mask, False, False))

    for mask in masks_top_down:
        reverted_pieces.append(general_diagonal_search(mover_pieces, opponent_pieces, selected_move, mask, True, False))
        reverted_pieces.append(general_diagonal_search(mover_pieces, opponent_pieces, selected_move, mask, False, True))

    return bitboard_handler.bitwise(reverted_pieces, "or")
