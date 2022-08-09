import random
import numpy as np
from collections import deque

from src.rl.mcts.monte_carlo_tree_search import MonteCarloTS
from src.rl.mcts.node import Node
from src.rl.mcts.prediction_queue import PredictionQueue
from src.rl.training.training_queue import TrainingQueue
from src.environment.board import Board
from src.environment.symmetries import BoardSymmetry

from src.rl.config import *
from src.environment.config import BOARD_SIZE

class SelfPlay:

    def __init__(self, prediction_queue, training_buffer, prediction_dict, n_iterations, thread_number):

        self.prediction_queue = prediction_queue
        self.training_buffer = training_buffer
        self.prediction_dict = prediction_dict
        self.n_iterations = n_iterations
        self.thread_number = thread_number

        self.current_node = None
        self.temperature = 1.0
        self.n_played_moves = 0

        self.encountered_nodes = []
        self.outcome = None

    def simulate_game(self):

        starting_position = Board(create_random_symmetry=USE_SYMMETRIES)
        self.current_node = Node(starting_position)

        while(not self.current_node.board.is_terminal):
            self.play_move()
            self.n_played_moves += 1
            self.update_temperature()

        self.outcome = self.current_node.board.reward

        self.send_samples_to_buffer()

    def send_samples_to_buffer(self):

        samples = []

        if USE_SYMMETRIES:
            symmetries = BoardSymmetry.symmetries
        else:
            symmetries = [BoardSymmetry.Operation.IDENTITY]

        for node, from_game in self.encountered_nodes:

            outcome_true = self.outcome if from_game else node.average_outcome

            search_policy = np.zeros(BOARD_SIZE*BOARD_SIZE)
            for index, value in zip(node.board.legal_moves["indices"], node.search_policy):
                search_policy[index] = value

            for symmetry in symmetries:

                symmetric_board = node.board.apply_symmetry(symmetry)

                white_pieces, black_pieces, turn, legal_moves, reward = symmetric_board.get_state(legal_moves_format="indices")
                board_inputs = np.stack([white_pieces, black_pieces, turn], axis=-1)
                masked_legal_moves = symmetric_board.legal_moves["array"]

                symmetric_search_policy = BoardSymmetry.symmetric(search_policy.tolist(),symmetry)
                symmetric_search_policy = np.array(symmetric_search_policy)

                inputs = [board_inputs, masked_legal_moves]
                outputs = [search_policy, outcome_true]

                samples.append([inputs, outputs])

        self.training_buffer.extend(samples)

    def update_temperature(self):

        if self.n_played_moves > N_MOVES_HIGHEST_TEMPERATURE:
            decreasing_factor = self.n_played_moves - N_MOVES_HIGHEST_TEMPERATURE + 1
            self.temperature = 1.0 / decreasing_factor

    def get_search_policy(self, node, temperature=None):

        if temperature is None:
            visits = [child.visit_count for child in node.children]

        else:
            exponent = 1.0/self.temperature
            visits = [pow(child.visit_count,exponent) for child in node.children]

        tot = float(sum(visits))
        search_policy = [visit/tot for visit in visits]

        return search_policy

    def get_first_discarded_node(self, children, chosen_node, simulation_policy):

        if len(children)==1:
            return None

        chosen_index = children.index(chosen_node)
        del children[chosen_index]
        del simulation_policy[chosen_index]

        if sum(simulation_policy)==0.0:
            return None

        second_choice = random.choices(children, weights=simulation_policy)[0]

        if second_choice.board.is_terminal or second_choice.visit_count<2:
            return None

        else:
            second_choice.search_policy = self.get_search_policy(second_choice)
            return second_choice

    def play_move(self):

        mcts = MonteCarloTS(self.current_node, self.prediction_queue, self.prediction_dict, self.n_iterations, self.thread_number)
        mcts.run_search()

        self.current_node.search_policy = self.get_search_policy(self.current_node)

        simulation_policy = self.get_search_policy(self.current_node, self.temperature)

        move = random.choices(self.current_node.children, weights=simulation_policy)[0]

        first_discarded = None
        if KEEP_TWO_NODES:
            first_discarded = self.get_first_discarded_node(self.current_node.children.copy(), move, simulation_policy.copy())

        self.encountered_nodes.append((self.current_node,True))
        self.current_node = move
        self.current_node.parent = None

        #Add a second node to encountered
        if first_discarded is not None and KEEP_TWO_NODES:
            self.encountered_nodes.append((first_discarded,False))
