import tensorflow as tf

from src.rl.training.learning_rate_schedule import LRSchedule

#Multiprocessing/Remote
LOCAL_WORKERS = 2
REMOTE_WORKERS = 2
USE_REMOTE = True
#DATA_MAX_SIZE = 1024*1024*512
HOST = "26.190.192.100"
PORT = 25600

#Paths
WEIGHTS_PATH = "weights"

#Architecture
CONV_FILTERS = 128
N_RESIDUAL_BLOCKS = 9
L2_REGULARIZER = tf.keras.regularizers.L2(l2=1e-4)

#Self-play
MCTS_ITERATIONS = {0:5,4:100,11:200}
N_MOVES_HIGHEST_TEMPERATURE = 12
CPUCT = 3 #To be set (3,4)?
EPS_DIRICHLET = 0.25
USE_SYMMETRIES = True
KEEP_TWO_NODES = True

#Training
TRAINING_QUEUE_LEN = 3840000 #To be set
TRAINING_POSITIONS = 100 #16384000
BATCH_SIZE = 1024 #To be set

EPOCHS_PER_GENERATION = 1 #2?
N_GAMES_BEFORE_TRAINING = 4
GOAL_GENERATION = 2

LEARNING_RATES = {0:0.003, 3:0.001, 10: 0.0001}
lr_schedule = LRSchedule(LEARNING_RATES)
OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, name="SGD")

#TODO: create separate matplotlib graph for drop values and adjust evaluation systems (average better?)
#TODO: Tune hyperparameters
#TODO: create distributed system
#TODO: Train
