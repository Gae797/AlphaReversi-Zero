from multiprocessing import Process, Queue, Manager
from collections import deque
import tensorflow as tf
import time
import os
import pickle

from src.rl.training.training_queue import TrainingQueue
from src.rl.mcts.prediction_queue import PredictionQueue
from src.rl.training.learning_rate_schedule import LRSchedule
from src.rl.mcts.selfplay import SelfPlay
import src.rl.architecture.network as network

from src.rl.config import *
from src.environment.config import BOARD_SIZE

class SelfPlayThread():

    def __init__(self, prediction_queue, workers_queue, training_buffer, prediction_dict, thread_number, depth):

        self.process = Process(target=SelfPlayThread.play_games, kwargs={"prediction_queue":prediction_queue,
                        "workers_queue":workers_queue,
                        "training_buffer":training_buffer,
                        "prediction_dict":prediction_dict,
                        "thread_number": thread_number,
                        "depth":depth})

    @staticmethod
    def play_games(prediction_queue, workers_queue, training_buffer, prediction_dict, thread_number, depth):

        workers_queue.put(thread_number)

        assert N_GAMES_BEFORE_TRAINING >= WORKERS

        n_games = N_GAMES_BEFORE_TRAINING // WORKERS
        completed_games = 0

        for i in range(n_games):
            selfplay = SelfPlay(prediction_queue, training_buffer, prediction_dict, depth, thread_number)
            selfplay.simulate_game()
            completed_games += 1
            print("Thread_{} has completed its game number {}".format(thread_number,completed_games))

        workers_queue.get()

class Trainer:

    def __init__(self):

        self.manager = Manager()
        games_buffer = self.manager.list()

        prediction_dict = {i+1:Queue(1) for i in range(WORKERS)}

        self.model, self.train_deque, self.completed_generations = self.load_checkpoint()

        self.training_queue = TrainingQueue(self.model, self.train_deque, games_buffer)
        self.prediction_queue = PredictionQueue(self.model,
                                                Queue(WORKERS),
                                                Queue(WORKERS),
                                                prediction_dict,
                                                2)

    def run(self):

        print("Training started")

        while(self.completed_generations != GOAL_GENERATION):
            start_time = time.time()
            self.run_selfplay_session()
            print("Self play session completed")
            self.run_training_session()
            self.completed_generations += 1
            self.save_checkpoint()
            print("Generation {} completed".format(self.completed_generations))
            print("--- %s seconds ---" % (time.time() - start_time))

        print("Training ended")

    def run_selfplay_session(self):

        for step in MCTS_ITERATIONS:
            if self.completed_generations >= step:
                depth = MCTS_ITERATIONS[step]

        threads = [SelfPlayThread(self.prediction_queue.queue,
                                self.prediction_queue.workers,
                                self.training_queue.buffer,
                                self.prediction_queue.prediction_dict,
                                i+1,
                                depth) for i in range(WORKERS)]

        for thread in threads:
            thread.process.start()

        self.prediction_queue.run_execution()

        for thread in threads:
            thread.process.join()

    def run_training_session(self):

        self.training_queue.train(TRAINING_POSITIONS // BATCH_SIZE)

    def load_last_generation(self):

        dirs = os.listdir(WEIGHTS_PATH)
        if len(dirs)==0:
            return None

        else:
            dirs.sort()
            return int(dirs[-1][-1])

    def save_checkpoint(self):

        path = os.path.join(WEIGHTS_PATH, "Generation {}".format(self.completed_generations))

        if not os.path.exists(path):
            os.makedirs(path)

        self.model.save_weights(os.path.join(path,"variables"))

        with open(os.path.join(path, "training_deque.pickle"), 'wb') as handle:
            pickle.dump(self.train_deque, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Checkpoint saved")

    def load_checkpoint(self):

        last_generation = self.load_last_generation()
        model = network.build_model(BOARD_SIZE, N_RESIDUAL_BLOCKS)

        if last_generation is None:
            completed_generations = 0
            train_deque = deque(maxlen=TRAINING_QUEUE_LEN)

            print("Created new model for training")

        else:
            path = os.path.join(WEIGHTS_PATH, "Generation {}".format(last_generation))
            model.load_weights(os.path.join(path, "variables"))

            with open(os.path.join(path, "training_deque.pickle"), 'rb') as handle:
                train_deque = pickle.load(handle)

            completed_generations = last_generation

            print("Loaded generation {}".format(last_generation))

        return model, train_deque, completed_generations
