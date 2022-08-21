'''
The prediction queue is used to estimate policies and values from multiple different
nodes (boards) at once, reducing the number of calls to the GPU and saving training time
'''

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

        self.model_graph = tf.function(lambda x: model(x)) #Execute network in fast graph mode

        self.prediction_dict = prediction_dict

    @property
    def passed_time(self):
        #Define how much time has passed from the last prediction
        return time.time()-self.timer

    def run_execution(self):

        self.timer = time.time()
        jobs_done = 0
        n_workers = self.workers.qsize()

        while(n_workers>0 or jobs_done==0):
            #Empty the queue and collect nodes
            if not self.queue.empty():
                pack = self.queue.get()
                self.collected_nodes.append(pack)

            n_workers = self.workers.qsize()
            size = len(self.collected_nodes)

            #Evaluate collected nodes if they are at least half of the workers
            #or if it has passed too much time from the last prediction
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
        legal_moves_batched = np.array(legal_moves_batched, dtype=np.float32)

        with tf.device('/device:GPU:0'):
            policies, values = self.model_graph([board_inputs_batched, legal_moves_batched])

        #Send predictions to respective workers
        for i, pack in enumerate(self.collected_nodes):
            thread_number = pack[1]
            prediction = [np.array(policies[i]), np.array(values[i])]
            self.prediction_dict[thread_number].put(prediction)

        self.collected_nodes.clear()
