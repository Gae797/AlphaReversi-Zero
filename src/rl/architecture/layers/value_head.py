'''
This class is the value head described in the AlphaGo Zero paper
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src.rl.config import *

class ValueHead(layers.Layer):

    def __init__(self):

        super(ValueHead, self).__init__()

        self.convolutional_layer = layers.Conv2D(32, #1?
                                                1,
                                                strides=1,
                                                padding="same",
                                                activation=None,
                                                use_bias=True,
                                                kernel_regularizer=L2_REGULARIZER)

        self.batch_norm_layer = layers.BatchNormalization()

        self.activation_layer = layers.ReLU()

        self.flatten_layer = layers.Flatten()
        self.dense_layer = layers.Dense(256, kernel_regularizer=L2_REGULARIZER)

        self.output_layer = layers.Dense(1, activation="tanh", kernel_regularizer=L2_REGULARIZER)

    def call(self, x):

        x = self.convolutional_layer(x)
        x = self.batch_norm_layer(x)
        x = self.activation_layer(x)

        x = self.flatten_layer(x)
        x = self.dense_layer(x)
        x = self.activation_layer(x)

        value = self.output_layer(x)

        return value
