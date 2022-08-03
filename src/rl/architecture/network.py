import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, metrics

from src.rl.architecture.layers.convolutional import ConvolutionalBlock
from src.rl.architecture.layers.residual import ResidualBlock
from src.rl.architecture.layers.policy_head import PolicyHead
from src.rl.architecture.layers.value_head import ValueHead

from src.rl.config import *

class AlphaReversiNetwork(tf.keras.Model):

    def __init__(self, n_squares, n_residual_blocks):

        super(AlphaReversiNetwork, self).__init__()

        self.n_squares = n_squares
        self.n_residual_blocks = n_residual_blocks

        self.convolutional_block = ConvolutionalBlock()
        self.residual_tower = [ResidualBlock() for _ in range(self.n_residual_blocks)]
        self.policy_head = PolicyHead(n_squares)
        self.value_head = ValueHead()

        self.policy_loss_tracker = metrics.Mean(name="policy_loss")
        self.value_loss_tracker = metrics.Mean(name="value_loss")
        self.total_loss_tracker = metrics.Mean(name="total_loss")

    def call(self, inputs):

        assert len(inputs)==2

        x = inputs[0]
        legal_moves = tf.cast(inputs[1], dtype=tf.float32)

        x = self.convolutional_block(x)

        for residual_block in self.residual_tower:
            x = residual_block(x)

        policy = self.policy_head([x, legal_moves])
        value = self.value_head(x)

        return policy, value

    @property
    def metrics(self):
        return [self.policy_loss_tracker, self.value_loss_tracker, self.total_loss_tracker]

    def train_step(self, data):

        x, y = data
        policy_true = y[0]
        value_true = tf.cast(y[1],dtype=tf.float32)

        with tf.GradientTape() as tape:
            policy_pred, value_pred = self(x, training=True)

            value_loss = tf.math.square(value_true - value_pred)
            policy_loss =  - tf.math.reduce_sum(policy_true * tf.math.log(policy_pred), axis=-1)

            #TODO transpose predictions

            total_value_loss = tf.math.reduce_sum(value_loss)
            total_policy_loss = tf.math.reduce_sum(policy_loss)

            loss = total_value_loss + total_policy_loss + self.losses

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.policy_loss_tracker.update_state(policy_loss)
        self.value_loss_tracker.update_state(value_loss)
        self.total_loss_tracker.update_state(loss)

        return {
            "policy_loss": self.policy_loss_tracker.result(),
            "value_loss": self.value_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
        }

def build_model(board_size, n_residual_blocks, verbose=True):

    n_squares = board_size*board_size

    board_inputs = tf.keras.Input(shape=(board_size, board_size, 3), name="board state")
    legal_moves_inputs = tf.keras.Input(shape=(n_squares), name="legal moves")

    inputs = [board_inputs, legal_moves_inputs]

    model = AlphaReversiNetwork(n_squares, n_residual_blocks)
    model(inputs)

    if verbose:
        model.summary()

    model.compile(optimizer=OPTIMIZER)

    return model
