import tensorflow as tf

from src.rl.training.learning_rate_schedule import LRSchedule

CONV_FILTERS = 128
N_MOVES_HIGHEST_TEMPERATURE = 20
CPUCT = 1 #To be set (3,4)?

TRAINING_QUEUE_LEN = 10000 #To be set
BATCH_SIZE = 256 #To be set
L2_REGULARIZER = tf.keras.regularizers.L2(l2=1e-4)

STARTING_LEARNING_RATE = 0.1
LR_DECREASING_STEPS = [] #To be set
lr_schedule = LRSchedule(STARTING_LEARNING_RATE, LR_DECREASING_STEPS)
OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, name="SGD")
