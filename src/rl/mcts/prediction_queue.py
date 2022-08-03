import queue
import numpy as np
import time

from src.rl.mcts.node import Node
import src.rl.architecture.network as network

class PredictionQueue:

    def __init__(self, model, max_await):

        self.queue = queue.Queue()
        self.collected_nodes = []

        self.run = False
        self.timer = 0.0

        self.n_parallel_calls = 0
        self.max_await = max_await
        self.model = model

    @property
    def passed_time(self):
        return time.time()-self.timer

    def start(self):

        self.run = True
        self.timer = time.time()

    def stop(self):

        self.run = False

    def add_node(self, node):

        self.queue.put(node)

    def add_worker(self):

        self.n_parallel_calls += 1

    def remove_worker(self):

        self.n_parallel_calls -= 1

    def run_execution(self):

        while(self.run):
            size = self.queue.qsize()
            if size > 0:

                if size >= self.n_parallel_calls:
                    nodes_to_be_collected = self.n_parallel_calls

                elif self.passed_time>self.max_await:
                    nodes_to_be_collected = size

                else:
                    nodes_to_be_collected = 0

                if nodes_to_be_collected > 0:
                    self.collect_nodes(nodes_to_be_collected)
                    self.evaluate_nodes()
                    self.timer = time.time()

    def collect_nodes(self, n):

        for i in range(n):
            node = self.queue.get()
            self.collected_nodes.append(node)

    def evaluate_nodes(self):

        board_inputs_batched = []
        legal_moves_batched = []
        for node in self.collected_nodes:
            white_pieces, black_pieces, turn, legal_moves, reward = node.board.get_state()

            board_inputs = np.stack([white_pieces, black_pieces, turn], axis=-1)

            board_inputs_batched.append(board_inputs)
            legal_moves_batched.append(legal_moves)

        board_inputs_batched = np.array(board_inputs_batched)
        legal_moves_batched = np.array(legal_moves_batched)

        policies, values = self.model.predict([board_inputs_batched, legal_moves_batched], batch_size=self.n_parallel_calls)

        for i, node in enumerate(self.collected_nodes):
            prediction = [policies[i], values[i]]
            node.set_estimation(prediction)

        self.collected_nodes.clear()
