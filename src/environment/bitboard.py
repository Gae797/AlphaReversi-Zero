'''
This module contains functions to create and manipulate bitboards: boards represented
as a binary string (or a number)
'''

import functools
import numpy as np

from src.environment.config import *

LENGTH = BOARD_SIZE*BOARD_SIZE

BITWISE_OPERATORS = ["and", "or", "xor"]

def value_to_string(value):

    #Cast an integer into a binary string of the board's length

    return format(value, '#0{}b'.format(LENGTH + 2))[:LENGTH+2]

def string_reformat(string_seq):

    #Pad a binary string with zeros to reach the board's length

    len_seq = len(string_seq)
    len_expected = LENGTH + 2

    diff = len_expected - len_seq

    if diff > 0:
        return string_seq + "0"*diff
    else:
        return string_seq

def print_bitboard(bitboard):

    #Print a bitboard as a matrix (using canonical Black's orientation)

    if not type(bitboard) is str:
        binary_string = value_to_string(bitboard)
    else:
        binary_string = bitboard

    binary_string = binary_string[2:]

    for i in range(BOARD_SIZE-1, -1, -1):
        print(binary_string[BOARD_SIZE*i:BOARD_SIZE*(i+1)])

    print()

def count_ones(bitboard):

    #Count number of "ones" inside a binary string

    binary_string = value_to_string(bitboard) if not type(bitboard) is str else bitboard
    binary_string = binary_string[2:]

    return binary_string.count("1")

def bitboard_to_numpy_array(bitboard):

    #Cast a binary string into a numpy array

    binary_string = value_to_string(bitboard) if not type(bitboard) is str else bitboard
    binary_string = binary_string[2:]

    return np.array(list(binary_string), dtype=int)

def bitboard_to_numpy_matrix(bitboard):

    #Cast a binary string into a numpy matrix (using canonical Black's orientation)

    binary_string = flip(bitboard, True)
    binary_string = binary_string[2:]

    array = np.array(list(binary_string), dtype=int)
    return np.reshape(array, (BOARD_SIZE, BOARD_SIZE))

#-------------------------------------------------------------------------------

def bitwise(bitboards, operator, return_as_string=False):

    #Apply a bitwise operator to map reducing a list of bitboards

    if operator not in BITWISE_OPERATORS:
        raise "Invalid bitwise operator: {}".format(operator)

    boards = [int(bitboard,2) if type(bitboard) is str else bitboard for bitboard in bitboards]

    if operator=="and":
        result = functools.reduce(lambda a, b: a&b, boards)

    elif operator=="or":
        result = functools.reduce(lambda a, b: a|b, boards)

    elif operator=="xor":
        result = functools.reduce(lambda a, b: a^b, boards)

    return value_to_string(result) if return_as_string else result

def complement(bitboard, return_as_string=False):

    #Compute the binary negation of a bitboard

    binary_value = int(bitboard, 2) if type(bitboard) is str else bitboard
    result = ~binary_value

    return value_to_string(result) if return_as_string else result

def negate(bitboard, return_as_string=True):

    #Negate a binary string by changing 0s into 1s and viceversa

    binary_value = value_to_string(bitboard) if not type(bitboard) is str else bitboard

    negate_table = binary_value.maketrans("01","10")
    result = "0b"+binary_value[2:].translate(negate_table)

    return result if return_as_string else int(result,2)

def shift(bitboard, steps, return_as_string=False):

    #Apply the binary shift operator to a bitboard

    binary_value = int(bitboard, 2)  if type(bitboard) is str else bitboard

    if steps==0:
        result = binary_value

    elif steps>0:
        result = binary_value>>steps

    else:
        result = binary_value<<abs(steps)

    return value_to_string(result) if return_as_string else result

#-------------------------------------------------------------------------------

def empty_bitboard():

    #Return the empty bitboard (all zeros)

    return value_to_string(0)

def full_bitboard():

    #Return the full bitboard (all ones)

    return value_to_string(1)

def starting_bitboard():

    #Generate the bitboard of the standard position (4 pieces in the centre)

    empty_squares = (BOARD_SIZE//2 - 1) * (BOARD_SIZE + 1)

    white_pieces = "0b" + "0"*(empty_squares+1) + "1" + "0"*(BOARD_SIZE-2) + "1" + "0"*(empty_squares+1)
    black_pieces = "0b" + "0"*empty_squares + "1" + "0"*BOARD_SIZE + "1" + "0"*empty_squares

    return white_pieces, black_pieces

#-------------------------------------------------------------------------------

def flip(bitboard, return_as_string=False):

    #Flip a given bitboard vertically

    if not type(bitboard) is str:
        binary_string = value_to_string(bitboard)
    else:
        binary_string = bitboard

    binary_string = binary_string[2:]

    flipped = "0b"
    for i in range(BOARD_SIZE-1, -1, -1):
        flipped+= binary_string[BOARD_SIZE*i:BOARD_SIZE*(i+1)]

    return flipped if return_as_string else int(flipped,2)

def mirror(bitboard, return_as_string=False):

    #Mirror a given bitboard horizontally

    if not type(bitboard) is str:
        binary_string = value_to_string(bitboard)
    else:
        binary_string = bitboard

    binary_string = binary_string[2:]

    mirrored = "0b"
    for i in range(BOARD_SIZE):
        row = binary_string[BOARD_SIZE*i:BOARD_SIZE*(i+1)]
        mirrored+= row[::-1]

    return mirrored if return_as_string else int(mirrored,2)

def rotate_90(bitboard, return_as_string=False):

    if not type(bitboard) is str:
        binary_string = value_to_string(bitboard)
    else:
        binary_string = bitboard

    binary_string = binary_string[2:]

    rotated = "0b"
    for i in range(BOARD_SIZE-1, -1, -1):
        col = binary_string[i::BOARD_SIZE]
        rotated+= col

    return rotated if return_as_string else int(rotated,2)

def rotate_270(bitboard, return_as_string=False):

    if not type(bitboard) is str:
        binary_string = value_to_string(bitboard)
    else:
        binary_string = bitboard

    binary_string = binary_string[2:]

    rotated = "0b"
    for i in range(BOARD_SIZE):
        col = binary_string[i::BOARD_SIZE]
        rotated+= col[::-1]

    return rotated if return_as_string else int(rotated,2)

def rotate_180(bitboard, return_as_string=False):

    flipped = flip(bitboard, True)
    return mirror(flipped, return_as_string)
