from threading import Thread

from src.rl.training.training_queue import training_queue
from src.rl.mcts.prediction_queue import PredictionQueue
from src.rl.mcts.selfplay import SelfPlay
import src.rl.architecture.network as network

from src.rl.config import *
from src.environment.config import BOARD_SIZE

class Trainer:

    def __init__(self):

        self.model = network.build_model(BOARD_SIZE, N_RESIDUAL_BLOCKS)
        self.training_queue = TrainingQueue(self.model, TRAINING_QUEUE_LEN)
        self.prediction_queue = PredictionQueue(self.model, WORKERS, 2)

        self.completed_generations = 0

        #TODO: load weights and training queue and generation

    def run(self):

        #TODO: save weights and training queue and generation

        while(self.completed_generations != GOAL_GENERATION):
            self.run_selfplay_session()
            self.run_training_session()

    def run_selfplay_session(self):

        threads = [Thread(self.play_games) for _ in range(WORKERS)]
        prediction_thread = Thread(self.prediction_queue.run_execution)

        self.prediction_queue.start()
        prediction_thread.start()

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        self.prediction_queue.stop()
        prediction_thread.join()

    def run_training_session(self):

        self.training_queue.train(TRAINING_STEPS_PER_GENERATION)

    def play_games(self):

        n_games = N_GAMES_BEFORE_TRAINING // WORKERS

        for i in range(n_games):
            selfplay = SelfPlay(self.prediction_queue, self.training_queue, MCTS_ITERATIONS)
            selfplay.simulate_game()
