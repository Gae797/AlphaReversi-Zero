'''
This module contains the operations that each worker has to do in order to play
the required number of games
'''

from multiprocessing import Process, Queue, Manager
from collections import deque

from src.rl.mcts.selfplay import SelfPlay

from src.rl.config import *

class SelfPlayThread():

    def __init__(self, prediction_queue, workers_queue, training_buffer,
                prediction_dict, thread_number, depth, games_to_play,
                n_workers, remote=False):

        self.process = Process(target=SelfPlayThread.play_games, kwargs={"prediction_queue":prediction_queue,
                        "workers_queue":workers_queue,
                        "training_buffer":training_buffer,
                        "prediction_dict":prediction_dict,
                        "thread_number": thread_number,
                        "depth":depth,
                        "games_to_play":games_to_play,
                        "n_workers":n_workers,
                        "remote":remote})

    @staticmethod
    def play_games(prediction_queue, workers_queue, training_buffer,
                prediction_dict, thread_number, depth, games_to_play,
                n_workers, remote):

        #Subscribe to the prediction queue
        workers_queue.put(thread_number)

        assert games_to_play >= n_workers

        #Assign the number of games to play for the worker
        n_games = games_to_play // n_workers
        completed_games = 0

        #Play the games
        for i in range(n_games):
            selfplay = SelfPlay(prediction_queue, training_buffer, prediction_dict, depth, thread_number, remote)
            selfplay.simulate_game()
            completed_games += 1
            print("Thread_{} has completed its game number {}".format(thread_number,completed_games))

        #Unsubscribe to the prediction queue
        workers_queue.get()
