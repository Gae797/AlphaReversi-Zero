import time
import platform
from subprocess import Popen, PIPE
from tempfile import TemporaryFile

from src.agents.agent_interface import AgentInterface
from src.environment import coordinate_handler

from src.environment.config import *

class EdaxAgent(AgentInterface):

    def __init__(self, depth, name="Edax Agent"):

        if platform.system() != "Windows":
            raise "Edax agent can be launched only on Windows systems"

        if BOARD_SIZE != 8:
            raise "Edax agent can be launched only on 8x8 boards"

        self.name = name

        self.depth = depth

        self.engine_started = False

    def play(self, board):

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
        move = coordinate_handler.coordinate_to_move(coordinate.lower())

        return move

    @property
    def is_external_engine(self):
        return True

    def start_new_game(self):

        self.stdout = TemporaryFile()

        init_params = [EDAX_ENGINE_PATH,
                        "eval-file", EDAX_EVAL_PATH,
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

        coordinate = coordinate_handler.move_to_coordinate(played_move)

        self.engine.stdin.write(coordinate+"\n")
        self.engine.stdin.flush()
        time.sleep(0.1)

    def force_pass(self):

        self.engine.stdin.write("ps\n")
        self.engine.stdin.flush()
        time.sleep(0.1)

    def force_sequence(self, sequence):

        for coordinate in sequence:
            self.engine.stdin.write(coordinate+"\n")
            self.engine.stdin.flush()
            time.sleep(0.2)
