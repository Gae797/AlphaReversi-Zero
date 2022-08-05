import numpy as np
import multiprocessing

from src.rl.mcts.node import Node
from src.rl.mcts.prediction_queue import PredictionQueue

from src.rl.config import *

import time

class MonteCarloTS():

    def __init__(self, root_node, prediction_queue, prediction_dict, n_iterations, thread_number):

        self.root = root_node
        self.prediction_queue = prediction_queue
        self.prediction_dict = prediction_dict
        self.n_iterations = n_iterations
        self.thread_number = thread_number

        if not self.root.is_expanded():
            self.root.expand()

        if not self.root.is_evaluated():
            self.evaluate(self.root)
            self.backup(self.root)

    def run_search(self):

        for i in range(self.n_iterations):
            self.run_iteration()

    def run_iteration(self):

        #TODO: add Dir noise if root and self-play

        selected_node = self.select(self.root)

        if not selected_node.is_evaluated():
            self.evaluate(selected_node)

        self.backup(selected_node)

    def select(self, node):

        if node.board.is_terminal or node.is_leaf():
            return node

        else:
            if not node.is_expanded():
                node.expand()

            select_policy = self.select_policy(node)
            selected_action = np.argmax(select_policy)
            selected_node = node.children[selected_action]

            return self.select(selected_node)

    def select_policy(self, parent_node):

        probabilities = parent_node.estimated_policy
        actions = parent_node.children

        select_policy = []
        for child_node, prior_prob in zip(actions, probabilities):
            uct = CPUCT * prior_prob * parent_node.visit_count / (child_node.visit_count+1)
            node_value = child_node.average_outcome if parent_node.board.turn else -child_node.average_outcome
            select_policy.append(uct + node_value)

        return select_policy

    def evaluate(self, node):

        pack = (node.board, self.thread_number)
        self.prediction_queue.put(pack)

        prediction = self.prediction_dict[self.thread_number].get()
        node.set_estimation(prediction)

    def backup(self, node):

        node.backup_update(node.estimated_value)
