'''
This class creates batches to train the network on
'''

import tensorflow as tf
import numpy as np
import random
import math

class Generator(tf.keras.utils.Sequence):

    def __init__(self, training_queue, batch_size, training_positions, shuffle=True):

        self.batch_size = batch_size

        #Create the training set by sampling the requeste number of positions from the training queue
        set_size = min(training_positions, len(training_queue))
        dataset = random.sample(training_queue, k=set_size)
        self.dataset = np.array(dataset, dtype=object)

        if shuffle:
            np.random.shuffle(self.dataset)

    def on_epoch_end(self):
        pass

    def __getitem__(self, idx):

        #Return next batch for training

        batch = self.dataset[idx * self.batch_size:(idx + 1) * self.batch_size]

        x, y = self.unpack_samples(batch)

        return x, y

    def __len__(self):

        return math.ceil(len(self.dataset) / self.batch_size)

    def unpack_samples(self, samples):

        #Extract features and labels from each sample of the training set

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
        legal_moves_batched = np.array(legal_moves_batched, dtype=np.float32)
        policy_outputs_batched = np.array(policy_outputs_batched)
        value_outputs_batched = np.array(value_outputs_batched)

        x_train = [board_inputs_batched, legal_moves_batched]
        y_train = [policy_outputs_batched, value_outputs_batched]

        return x_train, y_train
