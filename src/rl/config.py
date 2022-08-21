'''
This module contains all the hyperparameters and constants used for Reinforcement
Learning, for both training and playing
'''

import tensorflow as tf

from src.rl.training.learning_rate_schedule import LRSchedule

#Multiprocessing/Remote
LOCAL_WORKERS = 16 #Number of processes to be used for playing games on the local machine
REMOTE_WORKERS = 24 #Number of processes to be used for playing games on the remote machine
USE_REMOTE = True #Whether or not to use also the remote trainer
HOST = "25.67.231.114" #IP address to connect to
PORT = 25600 #Port to connect to

#Paths
WEIGHTS_PATH = "weights"

#Architecture
CONV_FILTERS = 128 #Number of convolutional filters used for all the layers
N_RESIDUAL_BLOCKS = 9 #Number of blocks in the residual tower
L2_REGULARIZER = tf.keras.regularizers.L2(l2=1e-4)

#Self-play
MCTS_ITERATIONS = {0:100,10:200} #Number of MCTS iterations from each completed generation on
N_MOVES_HIGHEST_TEMPERATURE = 30 #Number of moves for which the temperature is 1
CPUCT = 1 #Exploration rate for UCB in MCTS
EPS_DIRICHLET = 0.25 #Epsilon value for Dirichlet noise
USE_SYMMETRIES = True #If True the training set is augmented with all the possible symmetries
KEEP_TWO_NODES = True #If True a second node is extracted after each move is played during self-play

#Training
TRAINING_QUEUE_LEN = 3000000 #Length of the training window
TRAINING_POSITIONS = 819200 #Number of positions to sample from the training queue for each generation
BATCH_SIZE = 1024

EPOCHS_PER_GENERATION = 1 #Number of epochs of training for each generation
N_GAMES_BEFORE_TRAINING = 1000 #Number of games to be played before running the training session
GOAL_GENERATION = 25 #Number of generation required to stop the training process

LEARNING_RATES = {0:0.003, 10:0.001, 20: 0.0001} #Learning rates from each completed generation on
lr_schedule = LRSchedule(LEARNING_RATES)
OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, name="SGD")

#TODO: create separate matplotlib graph for drop values and adjust evaluation systems (average better?)
