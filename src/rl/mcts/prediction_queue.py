from multiprocessing import Queue
import numpy as np
import tensorflow as tf
import time

from src.rl.mcts.node import Node
import src.rl.architecture.network as network

class PredictionQueue:

    def __init__(self, model, queue, workers_queue, prediction_dict, max_await):

        self.queue = queue
        self.collected_nodes = []

        self.timer = 0.0

        self.workers = workers_queue
        self.max_await = max_await

        #self.model = model
        self.model_graph = tf.function(lambda x: model(x))

        self.prediction_dict = prediction_dict

    @property
    def passed_time(self):
        return time.time()-self.timer

    def run_execution(self):

        self.timer = time.time()
        jobs_done = 0
        n_workers = self.workers.qsize()

        while(n_workers>0 or jobs_done==0):
            if not self.queue.empty():
                pack = self.queue.get()
                self.collected_nodes.append(pack)

            n_workers = self.workers.qsize()
            size = len(self.collected_nodes)

            if size>0 and (size>=n_workers//2 or self.passed_time>self.max_await):
                self.evaluate_nodes()
                self.timer = time.time()
                jobs_done += 1

    def evaluate_nodes(self):

        board_inputs_batched = []
        legal_moves_batched = []
        for pack in self.collected_nodes:
            board = pack[0]

            white_pieces, black_pieces, turn, legal_moves, reward = board.get_state()

            board_inputs = np.stack([white_pieces, black_pieces, turn], axis=-1)

            board_inputs_batched.append(board_inputs)
            legal_moves_batched.append(legal_moves)

        board_inputs_batched = np.array(board_inputs_batched)
        legal_moves_batched = np.array(legal_moves_batched)

        with tf.device('/device:GPU:0'):
            policies, values = self.model_graph([board_inputs_batched, legal_moves_batched])

        for i, pack in enumerate(self.collected_nodes):
            thread_number = pack[1]
            prediction = [np.array(policies[i]), np.array(values[i])]
            self.prediction_dict[thread_number].put(prediction)

        self.collected_nodes.clear()
