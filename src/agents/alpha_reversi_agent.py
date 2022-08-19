'''
Main agent based on AlphaReversi trained network
'''

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

        self.evaluator = None

    def play(self, board):

        #Run Monte Carlo Tree Search witht the specified number of iterations
        root = Node(board)
        mcts = MonteCarloTS(root, None, None, self.depth, None, self.model)
        mcts.run_search()

        #Choose action with highest visit count
        visits = [child.visit_count for child in root.children]
        action = np.argmax(visits)

        move = board.legal_moves["indices"][action]

        #Update evaluation node drop
        if self.evaluator is not None:
            network_move_eval = root.children[action].average_outcome
            engine_move_eval = root.average_outcome
            self.evaluator.get_evaluations(network_move_eval, engine_move_eval)

        return move

    @property
    def is_external_engine(self):
        return False

    def attach_evaluator(self, evaluator):

        self.evaluator = evaluator

    def remove_evaluator(self):

        self.evaluator = None
