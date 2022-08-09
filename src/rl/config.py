import tensorflow as tf

from src.rl.training.learning_rate_schedule import LRSchedule

#Multiprocessing
WORKERS = 1

#Paths
WEIGHTS_PATH = "weights"

#Architecture
CONV_FILTERS = 128
N_RESIDUAL_BLOCKS = 9

#Self-play
MCTS_ITERATIONS = {0:100,4:200,11:400}
N_MOVES_HIGHEST_TEMPERATURE = 20
CPUCT = 1 #To be set (3,4)?
EPS_DIRICHLET = 0.25
USE_SYMMETRIES = True
KEEP_TWO_NODES = True

#Training
TRAINING_QUEUE_LEN = 16384000 #To be set
BATCH_SIZE = 1024 #To be set
L2_REGULARIZER = tf.keras.regularizers.L2(l2=1e-4)

EPOCHS_PER_STEP = 1 #2?
TRAINING_POSITIONS = 204800
N_GAMES_BEFORE_TRAINING = 1
GOAL_GENERATION = 1

LEARNING_RATES = {0:0.003, 3:0.001, 10: 0.0001}
lr_schedule = LRSchedule(LEARNING_RATES)
OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, name="SGD")
#OPTIMIZER = tf.keras.optimizers.Adam(lr_schedule)


#To be fixed:
#TODO: find solution for training (Generator or not?)
#TODO: create separate matplotlib graph for drop values and adjust evaluation systems (average better?)
#TODO: Tune hyperparameters
#TODO: run first test
#TODO: refactoring of code
#TODO: create distributed system
#TODO: Train

#Refactoring:
#TODO refactor configs
#TODO: refactor agents
#TODO: refactor board
#TODO: rafactor Trainer and SelfPlayThread
