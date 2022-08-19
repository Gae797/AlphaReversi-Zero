import tensorflow as tf

from src.rl.training.learning_rate_schedule import LRSchedule

#Multiprocessing/Remote
LOCAL_WORKERS = 16
REMOTE_WORKERS = 24
USE_REMOTE = True
HOST = "25.67.231.114"
PORT = 25600

#Paths
WEIGHTS_PATH = "weights"

#Architecture
CONV_FILTERS = 128
N_RESIDUAL_BLOCKS = 9
L2_REGULARIZER = tf.keras.regularizers.L2(l2=1e-4)

#Self-play
MCTS_ITERATIONS = {0:200,16:400}
N_MOVES_HIGHEST_TEMPERATURE = 16
CPUCT = 0.75
EPS_DIRICHLET = 0.25
USE_SYMMETRIES = True
KEEP_TWO_NODES = True

#Training
TRAINING_QUEUE_LEN = 3500000
TRAINING_POSITIONS = 819200
BATCH_SIZE = 1024

EPOCHS_PER_GENERATION = 1
N_GAMES_BEFORE_TRAINING = 1000
GOAL_GENERATION = 40

LEARNING_RATES = {0:0.002, 10:0.001, 20: 0.0002}
lr_schedule = LRSchedule(LEARNING_RATES)
OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, name="SGD")

#TODO: create separate matplotlib graph for drop values and adjust evaluation systems (average better?)
