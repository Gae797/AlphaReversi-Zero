import time
from subprocess import Popen, PIPE
from tempfile import TemporaryFile

from src.agents.agent_interface import AgentInterface

class EdaxAgent(AgentInterface):

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

    def __init__(self, depth, name="Edax Agent"):

        self.name = name

        self.depth = depth

        self.engine_started = False

    def play(self, board, timer):

        self.engine.stdin.write("go\n")
        self.engine.stdin.flush()

        if self.depth<20:
            time.sleep(1)
        else:
            time.sleep(3)

        self.stdout.seek(0)
        lines = self.stdout.readlines()
        last_line = lines[-2]

        coordinate = last_line[-4:-2].decode('UTF-8')
        move = self.coordinate_to_move(coordinate.lower())

        return move

    @property
    def is_external_engine(self):
        return True

    def start_new_game(self):

        self.stdout = TemporaryFile()

        init_params = ["edax engine/wEdax-x64.exe",
                        "eval-file", "edax engine/data/eval.dat",
                        "verbose", "0",
                        "book-usage", "on",
                        "book-randomness", "8",
                        "l", str(self.depth)]

        self.engine = Popen(init_params, stdin=PIPE, stdout=self.stdout, encoding='utf8')

        self.engine_started = True

    def close_game(self):

        self.engine.stdin.write("q")
        self.engine.stdin.flush()
        self.engine.terminate()
        self.engine.wait()

        self.stdout.close()

        self.engine_started = False

    def update_position(self, played_move):

        coordinate = self.move_to_coordinate(played_move)

        self.engine.stdin.write(coordinate+"\n")
        self.engine.stdin.flush()
        time.sleep(0.1)

    def force_pass(self):

        self.engine.stdin.write("ps\n")
        self.engine.stdin.flush()
        time.sleep(0.1)

    def move_to_coordinate(self, move):

        col = move % 8
        row = 8 - (move // 8)

        first_letter = EdaxAgent.move_translation_dict[col]
        second_letter = str(row)

        coordinate = first_letter+second_letter

        return coordinate

    def coordinate_to_move(self, coordinate):

        col = EdaxAgent.coordinate_translation_dict[coordinate[0]]
        row = int(coordinate[1])-1

        move = (7-row)*8 + col

        return move
