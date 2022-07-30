from src.agents.agent_interface import AgentInterface

import random

class RandomAgent(AgentInterface):

    def __init__(self, seed=None):

        if seed is not None:
            random.seed(seed)

    def play(self, board, time):

        legal_moves = board.legal_moves["indices"]
        move = random.choice(legal_moves)

        return move
