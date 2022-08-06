import tensorflow as tf

from src.rl.training.learning_rate_schedule import LRSchedule

WORKERS = 1

CONV_FILTERS = 128
N_RESIDUAL_BLOCKS = 9
MCTS_ITERATIONS = 100
N_MOVES_HIGHEST_TEMPERATURE = 20
CPUCT = 1 #To be set (3,4)?
EPS_DIRICHLET = 0.25
USE_SYMMETRIES = True

TRAINING_QUEUE_LEN = 16384000 #To be set
BATCH_SIZE = 128 #To be set
L2_REGULARIZER = tf.keras.regularizers.L2(l2=1e-4)

EPOCHS_PER_STEP = 1 #2?
TRAINING_POSITIONS = 204800
N_GAMES_BEFORE_TRAINING = 10
GOAL_GENERATION = 1

STARTING_LEARNING_RATE = 0.001
LR_DECREASING_STEPS = [] #To be set
#TODO: change depending just on current generation
lr_schedule = LRSchedule(STARTING_LEARNING_RATE, 0, LR_DECREASING_STEPS)
#OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, name="SGD")
OPTIMIZER = tf.keras.optimizers.Adam(lr_schedule)

WEIGHTS_PATH = "weights"

#Remainder:
#TODO gg2: multiple nodes from each simulation
#TODO gg2: dynamic MCTS
#TODO gg2: tune hyperparameters and optimizations
#TODO gg3: Edax agent
#TODO gg1: Match class
#TODO gg3: Evaluation by node value stability
#TODO gg3: Evaluation by matches with older versions or Edax (if succeded in implementing)
#TODO gg4: Test learning with dummy board
#TODO gg5: Create distributed system

#TODO: change training set?
#TODO: augment with symmetries?
