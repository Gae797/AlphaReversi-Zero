'''
This is the main module used by the client, for remote training.
It is used to play games that will be sent to the server
'''

import pickle
import socket
from multiprocessing import Process, Queue, Manager
from collections import deque

from src.rl.mcts.prediction_queue import PredictionQueue
from src.rl.training.player import SelfPlayThread
import src.rl.architecture.network as network

from src.rl.config import *
from src.environment.config import BOARD_SIZE

class RemoteTrainer:

    def __init__(self):

        self.manager = Manager()
        self.games_buffer = self.manager.list()

        self.completed_generations = 0
        self.games_to_play = 0

        self.model = network.build_model(BOARD_SIZE, N_RESIDUAL_BLOCKS)

        prediction_dict = {i+1:Queue(1) for i in range(REMOTE_WORKERS)}

        self.prediction_queue = PredictionQueue(self.model,
                                                Queue(REMOTE_WORKERS),
                                                Queue(REMOTE_WORKERS),
                                                prediction_dict,
                                                2)

        self.init_socket()

    def init_socket(self):

        #Create a connection with the server

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(None)
        self.socket.connect((HOST, PORT))

    def run(self):

        #Receive data, play games and send played positions

        while(self.completed_generations != GOAL_GENERATION):
            self.receive_data()
            self.run_selfplay_session()
            self.send_buffer()
            self.completed_generations += 1

        self.socket.close()

    def run_selfplay_session(self):

        #Define number of MCTS iterations depending on the current generation
        for step in MCTS_ITERATIONS:
            if self.completed_generations >= step:
                depth = MCTS_ITERATIONS[step]

        #Run a selfplay process for each of the workers
        threads = [SelfPlayThread(self.prediction_queue.queue,
                                self.prediction_queue.workers,
                                self.games_buffer,
                                self.prediction_queue.prediction_dict,
                                i+1,
                                depth,
                                self.games_to_play,
                                REMOTE_WORKERS,
                                True) for i in range(REMOTE_WORKERS)]

        for thread in threads:
            thread.process.start()

        self.prediction_queue.run_execution()

        #Wait for all workers to finish
        for thread in threads:
            thread.process.join()

    def send_buffer(self):

        #Send played positions to the server

        buffer = []
        buffer.extend(self.games_buffer)

        self.games_buffer[:] = []

        data = pickle.dumps(buffer)
        data_size = len(data).to_bytes(4,"big")

        self.socket.send(data_size)
        self.socket.send(data)

        print("Games sent")

    def receive_data(self):

        #Receive model's weights, completed generations number and number of games
        #to played from the server

        data_size = self.socket.recv(4)
        data_size = int.from_bytes(data_size,"big")

        data = bytearray()
        while len(data) < data_size:
            packet = self.socket.recv(data_size - len(data))
            data.extend(packet)

        print("Data received")

        #Set the correspondent variables with the received data
        unpacked_data = pickle.loads(data)
        weights = unpacked_data[0]
        self.completed_generations = unpacked_data[1]
        self.games_to_play = unpacked_data[2]

        self.model.set_weights(weights)
