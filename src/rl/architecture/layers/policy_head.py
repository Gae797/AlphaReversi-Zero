import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src.rl.config import *

class PolicyHead(layers.Layer):

    def __init__(self, n_squares):

        super(PolicyHead, self).__init__()

        self.convolutional_layer = layers.Conv2D(2, #32?
                                                1,
                                                strides=1,
                                                padding="same",
                                                activation=None,
                                                use_bias=True,
                                                kernel_regularizer=L2_REGULARIZER)

        self.batch_norm_layer = layers.BatchNormalization()

        self.activation_layer = layers.ReLU()

        self.flatten_layer = layers.Flatten()
        self.dense_layer = layers.Dense(n_squares, kernel_regularizer=L2_REGULARIZER)

        self.softmax_layer = layers.Softmax()

    def call(self, inputs):

        x = inputs[0]
        legal_moves = inputs[1]

        x = self.convolutional_layer(x)
        x = self.batch_norm_layer(x)
        x = self.activation_layer(x)

        x = self.flatten_layer(x)
        prob = self.dense_layer(x)

        prob = self.softmax_layer(prob)
        masked_prob = prob * legal_moves
        policy = self.softmax_layer(masked_prob)

        return policy
