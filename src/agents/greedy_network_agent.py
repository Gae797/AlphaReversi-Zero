'''
This agent is based on the main architecture of AlphaReversi but it chooses a move
just by using the prior probabilities in output from the network (so no MCTS)
'''

import numpy as np

from src.agents.agent_interface import AgentInterface
from src.environment.board import Board
import src.rl.architecture.network as network

class GreedyNetworkAgent(AgentInterface):

    def __init__(self, board_size, n_residual_blocks, weights, name= "Greedy Network Agent"):

        self.name = name

        self.model = network.build_model(board_size, n_residual_blocks)
        if weights is not None:
            self.model.load_weights(weights)

    def play(self, board):

        #Prepare the inputs
        white_pieces, black_pieces, turn, legal_moves, reward = board.get_state()

        board_inputs = np.stack([white_pieces, black_pieces, turn], axis=-1)
        batched_board_inputs = np.expand_dims(board_inputs, axis=0)
        batched_legal_moves = np.expand_dims(legal_moves, axis=0)

        #Predict
        policy, value = self.model([batched_board_inputs, batched_legal_moves], training=False)

        #Choose greedy action w.r.t. predicted policy
        move = np.argmax(policy)

        return move

    @property
    def is_external_engine(self):
        return False
