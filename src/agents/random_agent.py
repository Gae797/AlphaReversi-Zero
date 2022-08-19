'''
The random agent is the most basic one and it chooses a random move between the
legal ones
'''

from src.agents.agent_interface import AgentInterface

import random

class RandomAgent(AgentInterface):

    def __init__(self, seed=None, name="Random agent"):

        self.name = name

        if seed is not None:
            random.seed(seed)

    def play(self, board):

        #Pick a random move

        legal_moves = board.legal_moves["indices"]
        move = random.choice(legal_moves)

        return move

    @property
    def is_external_engine(self):
        return False
