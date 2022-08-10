from multiprocessing import Process, Queue, Manager
from collections import deque

from src.rl.mcts.prediction_queue import PredictionQueue
from src.rl.training.player import SelfPlayThread
import src.rl.architecture.network as network

from src.rl.config import *
from src.environment.config import BOARD_SIZE

class RemoteTrainer:

    def __init__(self, weights, completed_generations, games_buffer):

        self.games_buffer = games_buffer
        self.completed_generations = completed_generations

        self.model = network.build_model(BOARD_SIZE, N_RESIDUAL_BLOCKS)
        self.model.load_weights(weights)

        prediction_dict = {i+1:Queue(1) for i in range(WORKERS)}

        self.prediction_queue = PredictionQueue(self.model,
                                                Queue(WORKERS),
                                                Queue(WORKERS),
                                                prediction_dict,
                                                2)

    def run(self):

        for step in MCTS_ITERATIONS:
            if self.completed_generations >= step:
                depth = MCTS_ITERATIONS[step]

        threads = [SelfPlayThread(self.prediction_queue.queue,
                                self.prediction_queue.workers,
                                self.games_buffer,
                                self.prediction_queue.prediction_dict,
                                i+1,
                                depth) for i in range(WORKERS)]

        for thread in threads:
            thread.process.start()

        self.prediction_queue.run_execution()

        for thread in threads:
            thread.process.join()
