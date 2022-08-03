import queue
import numpy as np
import time

from src.rl.mcts.node import Node
import src.rl.architecture.network as network

class PredictionQueue:

    def __init__(self, model, n_parallel_calls, max_await):

        self.queue = queue.Queue()
        self.collected_nodes = []

        self.run = False
        self.timer = 0.0

        self.n_parallel_calls = n_parallel_calls
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

    def remove_worker(self):

        self.n_parallel_calls -= 1

    def run_execution(self):

        while(self.run):
            size = self.queue.qsize()

            if size >= self.n_parallel_calls:
                self.collect_nodes(self.n_parallel_calls)
                self.evaluate_nodes()
                self.timer = time.time()

            elif size>0 and self.passed_time>self.max_await:
                self.collect_nodes(size)
                self.evaluate_nodes()
                self.timer = time.time()

    def collect_nodes(self, n):

        for i in range(n):
            node = self.queue.get()
            self.collected_nodes.append(node)

    def evaluate_nodes(self):

        inputs = []
        for node in self.collect_nodes:
            white_pieces, black_pieces, turn, legal_moves, reward = node.board.get_state()

            board_inputs = np.stack([white_pieces, black_pieces, turn], axis=-1)

            inputs.append([board_inputs, legal_moves_inputs])

        inputs = np.array(inputs)

        predictions = self.model.predict(inputs, batch_size=self.n_parallel_calls)

        for i, node in enumerate(self.collected_nodes):
            prediction = predictions[i]
            node.set_estimation(prediction)

        self.collected_nodes.clear()
