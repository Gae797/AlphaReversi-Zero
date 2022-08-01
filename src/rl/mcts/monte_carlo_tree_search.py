import numpy as np

from src.rl.mcts.node import Node
from src.rl.mcts.prediction_queue import PredictionQueue

class MonteCarloTS():

    def __init__(self, root_node, prediction_queue, n_iterations):

        self.root = root_node
        self.prediction_queue = prediction_queue
        self.n_iterations = n_iterations

        if not self.root.is_expanded:
            self.root.expand()

        if not self.root.is_evaluated:
            self.evaluate(self.root)
            self.backup(self.root)

    def run_search(self):

        for i in range(self.n_iterations):
            self.run_iteration()

    def run_iteration(self):

        #TODO: add Dir noise if root and self-play

        selected_node = self.select(self.root)

        if not selected_node.is_evaluated:
            self.evaluate(selected_node)

        self.backup(selected_node)

    def select(self, node):

        if node.board.is_terminal or node.is_leaf():
            return node

        else:
            if not node.is_expanded:
                node.expand()

            search_policy = self.search_policy(node)
            selected_action = np.argmax(search_policy)
            selected_node = node.children[selected_action]

            return self.select(selected_node)

    def search_policy(self, parent_node):

        probabilities = parent_node.estimated_policy
        actions = parent_node.children

        search_policy = []
        for child_node, prior_prob in zip(actions, probabilities):
            uct = prior_prob * parent_node.visit_count / (child_node.visit_count+1)
            node_value = child_node.average_outcome if parent_node.board.turn else -child_node.average_outcome
            search_policy.append(uct + node_value)

        return search_policy

    def evaluate(self, node):

        prediction_queue.add_node(node)
        while(not node.is_evaluated):
            pass

    def backup(self, node):

        node.backup_update(node.estimated_value)
