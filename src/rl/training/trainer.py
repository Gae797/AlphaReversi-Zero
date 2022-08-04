from multiprocessing import Process, Queue
from collections import deque

from src.rl.training.training_queue import TrainingQueue
from src.rl.mcts.prediction_queue import PredictionQueue
from src.rl.mcts.selfplay import SelfPlay
import src.rl.architecture.network as network

from src.rl.config import *
from src.environment.config import BOARD_SIZE

class SelfPlayThread():

    def __init__(self, prediction_queue, workers_queue, training_queue, prediction_dict, thread_number):

        process_kwargs = {"prediction_queue":prediction_queue,
                        "workers_queue":workers_queue,
                        "training_queue":training_queue,
                        "prediction_dict":prediction_dict,
                        "thread_number": thread_number}

        self.process = Process(target=self.play_games, kwargs=process_kwargs)

    def play_games(self, prediction_queue, workers_queue, training_queue, prediction_dict, thread_number):

        workers_queue.put(thread_number)

        n_games = N_GAMES_BEFORE_TRAINING // WORKERS
        completed_games = 0

        for i in range(n_games):
            selfplay = SelfPlay(prediction_queue, training_queue, prediction_dict, MCTS_ITERATIONS, thread_number)
            selfplay.simulate_game()
            completed_games += 1
            print("Thread_{} has completed its game number {}".format(thread_number,completed_games))

        workers_queue.get()

class Trainer:

    def __init__(self):

        prediction_dict = {i+1:Queue(1) for i in range(WORKERS)}

        self.model = network.build_model(BOARD_SIZE, N_RESIDUAL_BLOCKS)
        self.training_queue = TrainingQueue(self.model, deque(maxlen=TRAINING_QUEUE_LEN))
        self.prediction_queue = PredictionQueue(self.model, Queue(WORKERS), Queue(WORKERS), prediction_dict, 2)

        self.completed_generations = 0

        #TODO: load weights and training queue and generation

    def run(self):

        #TODO: save weights and training queue and generation

        print("Training started")

        while(self.completed_generations != GOAL_GENERATION):
            self.run_selfplay_session()
            print("Self play session completed")
            self.run_training_session()
            self.completed_generations += 1
            print("Generation {} completed".format(self.completed_generations))

        print("Training ended")

    def run_selfplay_session(self):

        threads = [SelfPlayThread(self.prediction_queue.queue,
                                self.prediction_queue.workers,
                                self.training_queue.queue,
                                self.prediction_queue.prediction_dict,
                                i+1) for i in range(WORKERS)]

        for thread in threads:
            thread.process.start()

        self.prediction_queue.start()
        self.prediction_queue.run_execution()

        for thread in threads:
            thread.process.join()

        self.prediction_queue.stop()

    def run_training_session(self):

        self.training_queue.train(TRAINING_STEPS_PER_GENERATION)
