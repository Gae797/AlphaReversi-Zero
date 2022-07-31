import numpy as np

from src.agents.agent_interface import AgentInterface
from src.environment.board import Board
import src.rl.architecture.network as network

class GreedyNetworkAgent(AgentInterface):

    def __init__(self, board_size, n_residual_blocks):

        #TODO: load weights

        self.model = network.build_model(board_size, n_residual_blocks)

    def play(self, board, timer):

        white_pieces, black_pieces, turn, legal_moves, reward = board.get_state()

        board_inputs = np.stack([white_pieces, black_pieces, turn], axis=-1)
        batched_board_inputs = np.expand_dims(board_inputs, axis=0)

        outputs = self.model([batched_board_inputs, legal_moves])
        policy = outputs[0]

        move = np.argmax(policy)

        return move
