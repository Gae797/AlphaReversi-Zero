import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src.rl.config import *

class ConvolutionalBlock(layers.Layer):

    def __init__(self):

        super(ConvolutionalBlock, self).__init__()

        self.convolutional_layer = layers.Conv2D(CONV_FILTERS,
                                                3,
                                                strides=1,
                                                padding="same",
                                                activation=None,
                                                use_bias=True)

        self.batch_norm_layer = layers.BatchNormalization()

        self.activation_layer = layers.ReLU()

    def call(self, x):

        x = self.convolutional_layer(x)
        x = self.batch_norm_layer(x)
        x = self.activation_layer(x)

        return x
