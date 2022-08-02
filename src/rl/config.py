import tensorflow as tf

from src.rl.training.learning_rate_schedule import LRSchedule

WORKERS = 8

CONV_FILTERS = 128
N_RESIDUAL_BLOCKS = 9
MCTS_ITERATIONS = 400
N_MOVES_HIGHEST_TEMPERATURE = 20
CPUCT = 1 #To be set (3,4)?

TRAINING_QUEUE_LEN = 10000 #To be set
BATCH_SIZE = 128 #To be set
L2_REGULARIZER = tf.keras.regularizers.L2(l2=1e-4)

EPOCHS_PER_STEP = 1 #2?
TRAINING_STEPS_PER_GENERATION = 1000
N_GAMES_BEFORE_TRAINING = 800
GOAL_GENERATION = 20

STARTING_LEARNING_RATE = 0.1
LR_DECREASING_STEPS = [] #To be set
lr_schedule = LRSchedule(STARTING_LEARNING_RATE, LR_DECREASING_STEPS)
OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, name="SGD")

#Remainder:
#TODO: Trainer
#TODO: test extensively
#TODO: save/load weights and training queue and current generation
#TODO: Dir noise
#TODO: Resignation
#TODO: simmetries
#TODO: AlphaReversi agent
#TODO: handle repeated nodes in queue
#TODO: add locks and virtual losses?
#TODO: multiple nodes from each simulation
#TODO: Edax agent
#TODO: tune hyperparameters
#TODO: Match class
#TODO: Evaluation by node value stability
#TODO: Evaluation by matches with older versions or Edax (if succeded in implementing)
