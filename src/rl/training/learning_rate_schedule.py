'''
This class is used to decrease the learning rate accordingly to the current generation
'''

import tensorflow as tf

class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, lr_generation_schedule):

        self.lr_generation_schedule = lr_generation_schedule
        self.generation = None

    def set_generation(self, generation):

        self.generation = generation

    def __call__(self, step):

        for ref_generation in self.lr_generation_schedule:
            if self.generation >= ref_generation:
                learning_rate = self.lr_generation_schedule[ref_generation]

        return learning_rate
