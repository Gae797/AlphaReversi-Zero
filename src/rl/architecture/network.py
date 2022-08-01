import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src.rl.architecture.layers.convolutional import ConvolutionalBlock
from src.rl.architecture.layers.residual import ResidualBlock
from src.rl.architecture.layers.policy_head import PolicyHead
from src.rl.architecture.layers.value_head import ValueHead

class AlphaReversiNetwork(tf.keras.Model):

    def __init__(self, n_squares, n_residual_blocks):

        super(AlphaReversiNetwork, self).__init__()

        self.n_squares = n_squares
        self.n_residual_blocks = n_residual_blocks

        self.convolutional_block = ConvolutionalBlock()
        self.residual_tower = [ResidualBlock() for _ in range(self.n_residual_blocks)]
        self.policy_head = PolicyHead(n_squares)
        self.value_head = ValueHead()

    def call(self, inputs):

        assert len(inputs)==2

        x = inputs[0]
        legal_moves = tf.cast(inputs[1], dtype=tf.float32)

        x = self.convolutional_block(x)

        for residual_block in self.residual_tower:
            x = residual_block(x)

        policy = self.policy_head([x, legal_moves])
        value = self.value_head(x)

        return [policy, value]

    def train_step(self, data):

        x, y = data
        policy_true = y[0]
        value_true = y[1]

        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            policy_pred = predictions[0]
            value_pred = predictions[1]

            value_loss = tf.math.square(value_true - value_pred)
            policy_loss =  - tf.math.reduce_sum(policy_true * tf.math.log(policy_pred))
            loss = value_loss + policy_loss + self.losses

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        #TODO: return metrics?
        return None

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
