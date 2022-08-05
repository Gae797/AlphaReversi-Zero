import numpy as np
import random
from collections import deque

import src.rl.architecture.network as network

from src.rl.config import *

class TrainingQueue:

    def __init__(self, model, buffer, max_size):

        self.model = model
        self.queue = deque(maxlen=max_size)
        self.buffer = buffer

    def sample_queue(self):

        listed_queue = list(self.queue)
        samples = random.choices(listed_queue, k=BATCH_SIZE)

        return samples

    def unpack_samples(self, samples):

        x_train = []
        y_train = []

        board_inputs_batched = []
        legal_moves_batched = []
        policy_outputs_batched = []
        value_outputs_batched = []

        for sample in samples:
            inputs = sample[0]
            outputs = sample[1]

            board_inputs_batched.append(inputs[0])
            legal_moves_batched.append(inputs[1])

            policy_outputs_batched.append(outputs[0])
            value_outputs_batched.append(outputs[1])

        board_inputs_batched = np.array(board_inputs_batched)
        legal_moves_batched = np.array(legal_moves_batched)
        policy_outputs_batched = np.array(policy_outputs_batched)
        value_outputs_batched = np.array(value_outputs_batched)

        x_train = [board_inputs_batched, legal_moves_batched]
        y_train = [policy_outputs_batched, value_outputs_batched]

        return x_train, y_train

    def run_train_step(self):

        samples = self.sample_queue()

        x_train, y_train = self.unpack_samples(samples)

        self.model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS_PER_STEP)

    def train(self, steps):

        self.queue.extend(self.buffer)
        self.buffer[:] = [] #Clear the buffer

        for i in range(steps):
            self.run_train_step()
