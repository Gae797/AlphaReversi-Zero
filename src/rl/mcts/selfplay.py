import random
import numpy as np

from src.rl.mcts.monte_carlo_tree_search import MonteCarloTS
from src.rl.mcts.node import Node
from src.rl.mcts.prediction_queue import PredictionQueue
from src.rl.training.training_queue import TrainingQueue
from src.environment.board import Board

from src.rl.config import *
from src.environment.config import BOARD_SIZE

class SelfPlay:

    def __init__(self, prediction_queue, training_queue, n_iterations):

        self.prediction_queue = prediction_queue
        self.training_queue = training_queue
        self.n_iterations = n_iterations

        self.current_node = None
        self.temperature = 1.0
        self.n_played_moves = 0

        self.encountered_nodes = []
        self.outcome = None

    def simulate_game(self):

        #TODO: add resign

        starting_position = Board()
        self.current_node = Node(starting_position)

        while(not self.current_node.board.is_terminal):
            self.play_move()
            self.n_played_moves += 1
            self.update_temperature()

        self.outcome = self.current_node.board.reward

        self.send_samples_to_queue()

    def send_samples_to_queue(self):

        samples = []

        for node in self.encountered_nodes:

            white_pieces, black_pieces, turn, legal_moves, reward = node.board.get_state(legal_moves_format="indices")
            board_inputs = np.stack([white_pieces, black_pieces, turn], axis=-1)

            search_policy = np.zeros(BOARD_SIZE*BOARD_SIZE)
            for index, value in zip(legal_moves, node.search_policy):
                search_policy[index] = value

            outputs = [search_policy, self.outcome]

            samples.append([board_inputs, outputs])

        self.training_queue.add_samples(samples)

    def update_temperature(self):

        if self.n_played_moves > N_MOVES_HIGHEST_TEMPERATURE:
            decreasing_factor = self.n_played_moves - N_MOVES_HIGHEST_TEMPERATURE
            self.temperature = 1.0 / pow(decreasing_factor, 2)

    def play_move(self):

        mcts = MonteCarloTS(self.current_node, self.prediction_queue, self.n_iterations)
        mcts.run_search()

        self.current_node.search_policy = mcts.search_policy(self.current_node)

        exponent = 1.0/self.temperature
        visits = [pow(node.visit_count,exponent) for node in self.current_node.children]
        tot = sum(visits)
        visits = [visit/tot for visit in visits]

        move = random.choices(self.current_node.children, visits)[0]

        #TODO: add multiple nodes each time
        self.encountered_nodes.append(self.current_node)
        self.current_node = move
