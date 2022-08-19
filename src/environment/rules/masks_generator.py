'''
MasksGenerator takes care of building bitboards the represent all the possible
diagonals in the board
'''

import src.environment.bitboard as bitboard_handler

from src.environment.config import *

class MasksGenerator:

    masks_bottom_up = None
    masks_top_down = None

    @staticmethod
    def generate_mask_bottom_up(start, count):

        diagonal = "0b" + "0"*start
        for i in range(count-1):
            diagonal+= "1" + "0"*BOARD_SIZE
        diagonal+="1"

        diagonal = bitboard_handler.string_reformat(diagonal)

        return diagonal

    @staticmethod
    def generate_masks():

        masks_bottom_up = []
        masks_top_down = []

        for i in range(BOARD_SIZE-1):
            masks_bottom_up.append(MasksGenerator.generate_mask_bottom_up(i, BOARD_SIZE-i))
            if i!=0:
                masks_bottom_up.append(MasksGenerator.generate_mask_bottom_up(i*BOARD_SIZE, BOARD_SIZE-i))

        masks_top_down = [bitboard_handler.mirror(mask) for mask in masks_bottom_up]

        return masks_bottom_up, masks_top_down

    @staticmethod
    def get_masks():

        if MasksGenerator.masks_bottom_up is None or MasksGenerator.masks_top_down is None:
            MasksGenerator.masks_bottom_up, MasksGenerator.masks_top_down = MasksGenerator.generate_masks()

        return MasksGenerator.masks_bottom_up, MasksGenerator.masks_top_down
