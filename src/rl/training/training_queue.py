import numpy as np
import random
from collections import deque

import src.rl.architecture.network as network

from src.rl.config import *

class TrainingQueue:

    def __init__(self, model, size):

        self.model = model
        self.queue = deque(maxlen=size)

    def add_samples(self, samples):

        self.queue.extend(samples)

    def sample_queue(self):

        listed_queue = list(self.queue)
        samples = random.choices(listed_queue, k=BATCH_SIZE)

        return samples

    def split_samples(self, samples):

        x_train = []
        y_train = []

        for sample in samples:
            x_train.append(sample[0])
            y_train.append(sample[1])

        return x_train, y_train

    def run_train_step(self):

        samples = self.sample_queue()

        x_train, y_train = self.split_samples(samples)

        self.model.fit(x_train, y_train_ batch_size=BATCH_SIZE, epoch=1)

    def train(self, epochs):

        for i in range(epochs):
            self.run_train_step()
