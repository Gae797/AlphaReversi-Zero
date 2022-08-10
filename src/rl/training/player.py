from multiprocessing import Process, Queue, Manager
from collections import deque

from src.rl.mcts.selfplay import SelfPlay

from src.rl.config import *

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
