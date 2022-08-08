coordinate_translation_dict = {
"a":0,
"b":1,
"c":2,
"d":3,
"e":4,
"f":5,
"g":6,
"h":7
}

move_translation_dict = {
0:"a",
1:"b",
2:"c",
3:"d",
4:"e",
5:"f",
6:"g",
7:"h"
}

def move_to_coordinate(move):

    col = move % 8
    row = 8 - (move // 8)

    first_letter = move_translation_dict[col]
    second_letter = str(row)

    coordinate = first_letter+second_letter

    return coordinate

def coordinate_to_move(coordinate):

    col = coordinate_translation_dict[coordinate[0]]
    row = int(coordinate[1])-1

    move = (7-row)*8 + col

    return move

def convert_sequence(sequence):

    return [coordinate_to_move(coordinate) for coordinate in sequence]
