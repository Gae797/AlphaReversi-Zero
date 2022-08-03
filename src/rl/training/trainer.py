import threading

from src.rl.training.training_queue import TrainingQueue
from src.rl.mcts.prediction_queue import PredictionQueue
from src.rl.mcts.selfplay import SelfPlay
import src.rl.architecture.network as network

from src.rl.config import *
from src.environment.config import BOARD_SIZE

class SelfPlayThread(threading.Thread):

    def __init__(self, prediction_queue, training_queue, *args, **kwargs):

        super(SelfPlayThread,self).__init__(*args, **kwargs)
        self.prediction_queue = prediction_queue
        self.training_queue = training_queue

    def play_games(self):

        self.prediction_queue.add_worker()

        n_games = N_GAMES_BEFORE_TRAINING // WORKERS

        for i in range(n_games):
            selfplay = SelfPlay(self.prediction_queue, self.training_queue, MCTS_ITERATIONS)
            selfplay.simulate_game()

        self.prediction_queue.remove_worker()

    def run(self):

        self.play_games()

class PredictionQueueThread(threading.Thread):

    def __init__(self, prediction_queue, *args, **kwargs):

        super(PredictionQueueThread,self).__init__(*args, **kwargs)
        self.prediction_queue = prediction_queue

    def run(self):

        self.prediction_queue.run_execution()

class Trainer:

    def __init__(self):

        self.model = network.build_model(BOARD_SIZE, N_RESIDUAL_BLOCKS)
        self.training_queue = TrainingQueue(self.model, TRAINING_QUEUE_LEN)
        self.prediction_queue = PredictionQueue(self.model, 2)

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

        threads = [SelfPlayThread(self.prediction_queue, self.training_queue) for _ in range(WORKERS)]

        self.prediction_queue.start()
        prediction_thread = PredictionQueueThread(self.prediction_queue)
        prediction_thread.start()

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        self.prediction_queue.stop()
        prediction_thread.join()

    def run_training_session(self):

        self.training_queue.train(TRAINING_STEPS_PER_GENERATION)
