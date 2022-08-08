import numpy as np

from src.agents.agent_interface import AgentInterface
from src.environment.board import Board
import src.rl.architecture.network as network
from src.rl.mcts.monte_carlo_tree_search import MonteCarloTS
from src.rl.mcts.node import Node

class AlphaReversiAgent(AgentInterface):

    def __init__(self, board_size, n_residual_blocks, weights, mcts_depth, name="AlphaReversi Agent"):

        self.name = name

        self.model = network.build_model(board_size, n_residual_blocks)
        if weights is not None:
            self.model.load_weights(weights)

        self.depth = mcts_depth

    def play(self, board, timer):

        root = Node(board)
        mcts = MonteCarloTS(root, None, None, self.depth, None, self.model)
        mcts.run_search()

        visits = [child.visit_count for child in root.children]
        action = np.argmax(visits)

        move = board.legal_moves["indices"][action]

        return move

    @property
    def is_external_engine(self):
        return False
