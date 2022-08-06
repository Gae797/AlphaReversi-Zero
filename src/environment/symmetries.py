import src.environment.bitboard as bitboard_handler
from enum import Enum
import random

class BoardSymmetry:

    Operation = Enum("Operation", "IDENTITY ROT_90 ROT_180 ROT_270 ROT_DIAG_1 ROT_DIAG_2 FLIP MIRROR RANDOM")

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
    def identity(bitboard):
        return bitboard

    @staticmethod
    def rotate_90(bitboard):
        return bitboard_handler.rotate_90(bitboard, return_as_string=True)

    @staticmethod
    def rotate_180(bitboard):
        return bitboard_handler.rotate_180(bitboard, return_as_string=True)

    @staticmethod
    def rotate_270(bitboard):
        return bitboard_handler.rotate_270(bitboard, return_as_string=True)

    @staticmethod
    def flip(bitboard):
        return bitboard_handler.flip(bitboard, return_as_string=True)

    @staticmethod
    def mirror(bitboard):
        return bitboard_handler.mirror(bitboard, return_as_string=True)

    @staticmethod
    def rotate_diagonal_1(bitboard):
        return BoardSymmetry.rotate_90(BoardSymmetry.flip(bitboard))

    @staticmethod
    def rotate_diagonal_2(bitboard):
        return BoardSymmetry.rotate_270(BoardSymmetry.flip(bitboard))
