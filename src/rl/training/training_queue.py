import numpy as np
import random
from collections import deque

import src.rl.architecture.network as network
from src.rl.training.training_set_generator import Generator

from src.rl.config import *

class TrainingQueue:

    def __init__(self, model, queue, buffer):

        self.model = model
        self.queue = queue
        self.buffer = buffer

    def train(self):

        self.queue.extend(self.buffer)
        self.buffer[:] = [] #Clear the buffer

        training_generator = Generator(list(self.queue), BATCH_SIZE, TRAINING_POSITIONS)

        self.model.fit(training_generator, epochs=EPOCHS_PER_GENERATION)
