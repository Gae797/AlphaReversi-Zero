from multiprocessing import Process, Queue, Manager
from collections import deque
import tensorflow as tf
import time
import os
import pickle
import socket

from src.rl.training.training_queue import TrainingQueue
from src.rl.mcts.prediction_queue import PredictionQueue
from src.rl.training.learning_rate_schedule import LRSchedule
from src.rl.training.player import SelfPlayThread
from src.environment.board import Board
from src.environment.symmetries import BoardSymmetry
import src.rl.architecture.network as network

from src.rl.config import *
from src.environment.config import BOARD_SIZE

class Trainer:

    def __init__(self):

        self.manager = Manager()
        games_buffer = self.manager.list()

        prediction_dict = {i+1:Queue(1) for i in range(LOCAL_WORKERS)}

        self.model, self.train_deque, self.completed_generations = self.load_checkpoint()

        self.training_queue = TrainingQueue(self.model, self.train_deque, games_buffer)
        self.prediction_queue = PredictionQueue(self.model,
                                                Queue(LOCAL_WORKERS),
                                                Queue(LOCAL_WORKERS),
                                                prediction_dict,
                                                2)

        if USE_REMOTE:
            self.init_socket()

    def init_socket(self):

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((HOST, PORT))

        self.socket.listen(1)
        self.conn, self.address = self.socket.accept()

    def run(self):

        print("Training started")

        while(self.completed_generations != GOAL_GENERATION):
            start_time = time.time()

            self.run_selfplay_session()
            print("Self play session completed")

            self.run_training_session()
            self.completed_generations += 1
            lr_schedule.set_generation(self.completed_generations)
            self.save_checkpoint()

            print("Generation {} completed".format(self.completed_generations))
            print("--- %s seconds ---" % (time.time() - start_time))

        print("Training ended")

        if USE_REMOTE:
            self.socket.close()

    def run_selfplay_session(self):

        if USE_REMOTE:
            games_per_worker = N_GAMES_BEFORE_TRAINING // (LOCAL_WORKERS + REMOTE_WORKERS)
            n_local_games = games_per_worker * LOCAL_WORKERS
            n_remote_games = games_per_worker * REMOTE_WORKERS

            self.send_data(n_remote_games)

        else:
            n_local_games = N_GAMES_BEFORE_TRAINING // LOCAL_WORKERS

        for step in MCTS_ITERATIONS:
            if self.completed_generations >= step:
                depth = MCTS_ITERATIONS[step]

        threads = [SelfPlayThread(self.prediction_queue.queue,
                                self.prediction_queue.workers,
                                self.training_queue.buffer,
                                self.prediction_queue.prediction_dict,
                                i+1,
                                depth,
                                n_local_games,
                                LOCAL_WORKERS) for i in range(LOCAL_WORKERS)]

        for thread in threads:
            thread.process.start()

        self.prediction_queue.run_execution()

        for thread in threads:
            thread.process.join()

        if USE_REMOTE:
            self.receive_buffer()

    def run_training_session(self):

        self.training_queue.train()

    def load_last_generation(self):

        dirs = os.listdir(WEIGHTS_PATH)
        if len(dirs)==0:
            return 0

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

        completed_generations = self.load_last_generation()

        lr_schedule.set_generation(completed_generations)
        model = network.build_model(BOARD_SIZE, N_RESIDUAL_BLOCKS)

        if completed_generations==0:
            train_deque = deque(maxlen=TRAINING_QUEUE_LEN)

            print("Created new model for training")

        else:
            path = os.path.join(WEIGHTS_PATH, "Generation {}".format(completed_generations))
            model.load_weights(os.path.join(path, "variables"))

            with open(os.path.join(path, "training_deque.pickle"), 'rb') as handle:
                train_deque = pickle.load(handle)

            print("Loaded generation {}".format(completed_generations))

        return model, train_deque, completed_generations

    def send_data(self, n_required_games):

        weights = self.model.get_weights()
        n_completed_generations = self.completed_generations

        data = pickle.dumps([weights, n_completed_generations, n_required_games])
        data_size = len(data).to_bytes(4,"big")

        self.conn.send(data_size)
        self.conn.send(data)

        print("Data sent")

    def receive_buffer(self):

        buffer_size = self.conn.recv(4)
        buffer_size = int.from_bytes(buffer_size,"big")

        buffer = bytearray()
        while len(buffer) < buffer_size:
            packet = self.conn.recv(buffer_size - len(buffer))
            buffer.extend(packet)

        print("Games received")

        unpacked_buffer = pickle.loads(buffer)

        self.send_remote_games_to_trainer_queue(unpacked_buffer)

    def send_remote_games_to_trainer_queue(self, buffer):

        samples = []

        if USE_SYMMETRIES:
            symmetries = BoardSymmetry.symmetries
        else:
            symmetries = [BoardSymmetry.Operation.IDENTITY]

        for node_board, node_search_policy, node_average_outcome, from_game, outcome in buffer:

            outcome_true = outcome if from_game else node_average_outcome

            search_policy = np.zeros(BOARD_SIZE*BOARD_SIZE)
            for index, value in zip(node_board.legal_moves["indices"], node_search_policy):
                search_policy[index] = value

            for symmetry in symmetries:

                symmetric_board = node_board.apply_symmetry(symmetry)

                white_pieces, black_pieces, turn, legal_moves, reward = symmetric_board.get_state(legal_moves_format="indices")
                board_inputs = np.stack([white_pieces, black_pieces, turn], axis=-1)
                masked_legal_moves = symmetric_board.legal_moves["array"]

                symmetric_search_policy = BoardSymmetry.symmetric(search_policy.tolist(),symmetry)
                symmetric_search_policy = np.array(symmetric_search_policy)

                inputs = [board_inputs, masked_legal_moves]
                outputs = [symmetric_search_policy, outcome_true]

                samples.append([inputs, outputs])

        self.training_buffer.extend(samples)
