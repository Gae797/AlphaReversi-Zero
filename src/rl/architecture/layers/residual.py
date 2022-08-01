import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src.rl.config import *

class ResidualBlock(layers.Layer):

    def __init__(self):

        super(ResidualBlock, self).__init__()

        self.convolutional_layer_1 = layers.Conv2D(CONV_FILTERS,
                                                3,
                                                strides=1,
                                                padding="same",
                                                activation=None,
                                                use_bias=True,
                                                kernel_regularizer=L2_REGULARIZER)

        self.convolutional_layer_2 = layers.Conv2D(CONV_FILTERS,
                                                3,
                                                strides=1,
                                                padding="same",
                                                activation=None,
                                                use_bias=True,
                                                kernel_regularizer=L2_REGULARIZER)

        self.batch_norm_layer = layers.BatchNormalization()

        self.activation_layer = layers.ReLU()

    def call(self, x):

        f_x = self.convolutional_layer_1(x)
        f_x = self.batch_norm_layer(f_x)
        f_x = self.activation_layer(f_x)

        f_x = self.convolutional_layer_2(f_x)
        f_x = self.batch_norm_layer(f_x)
        f_x = f_x + x
        f_x = self.activation_layer(f_x)

        return f_x
