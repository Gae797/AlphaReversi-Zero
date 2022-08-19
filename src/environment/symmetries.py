'''
This module contains methods to manipulate a bitboard and transform it into one
of the 8 possible symmetric positions
'''

from enum import Enum
import random

import src.environment.bitboard as bitboard_handler
from src.environment.config import *

class BoardSymmetry:

    Operation = Enum("Operation", "IDENTITY ROT_90 ROT_180 ROT_270 ROT_DIAG_1 ROT_DIAG_2 FLIP MIRROR RANDOM")

    symmetries = [
    Operation.IDENTITY,
    Operation.ROT_90,
    Operation.ROT_180,
    Operation.ROT_270,
    Operation.ROT_DIAG_1,
    Operation.ROT_DIAG_2,
    Operation.FLIP,
    Operation.MIRROR
    ]

    @staticmethod
    def symmetric(bitboard, operation):

        if operation==BoardSymmetry.Operation.IDENTITY:
            return BoardSymmetry.identity(bitboard)

        elif operation==BoardSymmetry.Operation.ROT_90:
            return BoardSymmetry.rotate_90(bitboard)

        elif operation==BoardSymmetry.Operation.ROT_180:
            return BoardSymmetry.rotate_180(bitboard)

        elif operation==BoardSymmetry.Operation.ROT_270:
            return BoardSymmetry.rotate_270(bitboard)

        elif operation==BoardSymmetry.Operation.ROT_DIAG_1:
            return BoardSymmetry.rotate_diagonal_1(bitboard)

        elif operation==BoardSymmetry.Operation.ROT_DIAG_2:
            return BoardSymmetry.rotate_diagonal_2(bitboard)

        elif operation==BoardSymmetry.Operation.FLIP:
            return BoardSymmetry.flip(bitboard)

        elif operation==BoardSymmetry.Operation.MIRROR:
            return BoardSymmetry.mirror(bitboard)

        elif operation==BoardSymmetry.Operation.RANDOM:
            random_operation = random.randint(1, 8)
            return BoardSymmetry.symmetric(bitboard, BoardSymmetry.Operation(random_operation))

    @staticmethod
    def random_symmetry(bitboard):
        random_operation = random.randint(1, 8)
        symmetry = BoardSymmetry.symmetric(bitboard, BoardSymmetry.Operation(random_operation))

        return symmetry, random_operation

    @staticmethod
    def identity(bitboard):
        return bitboard

    @staticmethod
    def rotate_90(bitboard):

        if type(bitboard) is list:
            rotated = []
            for i in range(BOARD_SIZE-1, -1, -1):
                col = bitboard[i::BOARD_SIZE]
                rotated+= col

            return rotated

        else:
            return bitboard_handler.rotate_90(bitboard, return_as_string=True)

    @staticmethod
    def rotate_180(bitboard):

        if type(bitboard) is list:
            flipped = BoardSymmetry.flip(bitboard)
            return BoardSymmetry.mirror(bitboard)

        else:
            return bitboard_handler.rotate_180(bitboard, return_as_string=True)

    @staticmethod
    def rotate_270(bitboard):

        if type(bitboard) is list:
            rotated = []
            for i in range(BOARD_SIZE):
                col = bitboard[i::BOARD_SIZE]
                rotated+= col[::-1]

            return rotated

        else:
            return bitboard_handler.rotate_270(bitboard, return_as_string=True)

    @staticmethod
    def flip(bitboard):

        if type(bitboard) is list:
            flipped = []
            for i in range(BOARD_SIZE-1, -1, -1):
                flipped+= bitboard[BOARD_SIZE*i:BOARD_SIZE*(i+1)]

            return flipped

        else:
            return bitboard_handler.flip(bitboard, return_as_string=True)

    @staticmethod
    def mirror(bitboard):

        if type(bitboard) is list:
            mirrored = []
            for i in range(BOARD_SIZE):
                row = bitboard[BOARD_SIZE*i:BOARD_SIZE*(i+1)]
                mirrored+= row[::-1]

            return mirrored

        else:
            return bitboard_handler.mirror(bitboard, return_as_string=True)

    @staticmethod
    def rotate_diagonal_1(bitboard):

        return BoardSymmetry.rotate_90(BoardSymmetry.flip(bitboard))

    @staticmethod
    def rotate_diagonal_2(bitboard):

        return BoardSymmetry.rotate_270(BoardSymmetry.flip(bitboard))
