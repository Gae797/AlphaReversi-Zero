'''
This module manages the algorithm behind the Monte Carlo Tree Search
'''

import numpy as np
import math
import multiprocessing

from src.rl.mcts.node import Node
from src.rl.mcts.prediction_queue import PredictionQueue

from src.rl.config import *

import time

class MonteCarloTS():

    def __init__(self, root_node, prediction_queue, prediction_dict, n_iterations, thread_number, local_model=None):

        self.root = root_node
        self.prediction_queue = prediction_queue
        self.prediction_dict = prediction_dict
        self.n_iterations = n_iterations
        self.thread_number = thread_number
        self.local_model = local_model

        if not self.root.is_expanded():
            self.root.expand()

        if not self.root.is_evaluated():
            self.evaluate(self.root)
            self.backup(self.root)

        self.add_dirichlet_noise(self.root)

    def run_search(self):

        #Run the MCTS for the specified number of iterations

        for i in range(self.n_iterations):
            self.run_iteration()

    def run_iteration(self):

        #3 steps: select node (leaf), evaluate and backup

        selected_node = self.select(self.root)

        if not selected_node.is_evaluated():
            self.evaluate(selected_node)

        self.backup(selected_node)

    def add_dirichlet_noise(self, node):

        #Choose alpha
        n_actions = len(node.children)
        alpha = min(1, 10.0/n_actions)
        alpha_vector = [alpha] * n_actions

        #Sample from dirichlet distribution
        dirichlet_noise = np.random.dirichlet(alpha_vector)

        #Apply noise with epsilon weight
        node.estimated_policy = (1-EPS_DIRICHLET)*node.estimated_policy + EPS_DIRICHLET*dirichlet_noise

    def select(self, node):

        #Return node if it is a leaf
        if node.board.is_terminal or node.is_leaf():
            return node

        else:
            if not node.is_expanded():
                node.expand()

            #Compute tree policy (UCB)
            select_policy = self.select_policy(node)
            selected_action = np.argmax(select_policy)
            selected_node = node.children[selected_action]

            return self.select(selected_node)

    def select_policy(self, parent_node):

        probabilities = parent_node.estimated_policy
        actions = parent_node.children

        #Compute UCB for each action
        select_policy = []
        for child_node, prior_prob in zip(actions, probabilities):
            ucb = CPUCT * prior_prob * (math.sqrt(parent_node.visit_count) / (child_node.visit_count+1))
            node_value = child_node.average_outcome if parent_node.board.turn else -child_node.average_outcome
            select_policy.append(ucb + node_value)

        return select_policy

    def evaluate(self, node):

        #Ask prediction queue to evaluate if no local model is given (training)
        if self.local_model is None:
            #Send request to prediction queue
            pack = (node.board, self.thread_number)
            self.prediction_queue.put(pack)

            #Get answer from prediction queue
            prediction = self.prediction_dict[self.thread_number].get()
            node.set_estimation(prediction)

        #Do a local prediction if a local model is given (real game)
        else:
            white_pieces, black_pieces, turn, legal_moves, reward = node.board.get_state()

            board_inputs = np.stack([white_pieces, black_pieces, turn], axis=-1)
            batched_board_inputs = np.expand_dims(board_inputs, axis=0)
            batched_legal_moves = np.expand_dims(legal_moves, axis=0)

            policy, value = self.local_model([batched_board_inputs, batched_legal_moves], training=False)
            prediction = [np.array(policy[0]), np.array(value[0])]
            node.set_estimation(prediction)

    def backup(self, node):

        node.backup_update(node.estimated_value)
