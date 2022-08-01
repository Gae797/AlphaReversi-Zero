import random

from src.rl.mcts.monte_carlo_tree_search import MonteCarloTS
from src.rl.mcts.node import Node
from src.rl.mcts.prediction_queue import PredictionQueue
from src.environment.board import Board

from src.rl.config import *

class SelfPlay:

    def __init__(self, prediction_queue, n_iterations):

        self.prediction_queue = prediction_queue
        self.n_iterations = n_iterations

        self.current_node = None
        self.temperature = 1.0
        self.n_played_moves = 0

        self.encountered_nodes = []
        self.outcome = None

    def simulate_game(self):

        starting_position = Board()
        self.current_node = Node(starting_position)

        while(not self.current_node.board.is_terminal):
            self.play_move()
            self.n_played_moves += 1
            self.update_temperature()

        self.outcome = self.current_node.board.reward

    def update_temperature(self):

        if self.n_played_moves > N_MOVES_HIGHEST_TEMPERATURE:
            decreasing_factor = self.n_played_moves - N_MOVES_HIGHEST_TEMPERATURE
            self.temperature = 1.0 / pow(decreasing_factor, 2)

    def play_move(self):

        mcts = MonteCarloTS(self.current_node, self.prediction_queue, self.n_iterations)
        mcts.run_search()

        exponent = 1.0/self.temperature
        visits = [pow(node.visit_count,exponent) for node in self.current_node.children]
        tot = sum(visits)
        visits = [visit/tot for visit in visits]

        move = random.choice(self.current_node.children, visits)

        #TODO: add multiple nodes each time
        self.encountered_nodes.append(self.current_node)
        self.current_node = move
