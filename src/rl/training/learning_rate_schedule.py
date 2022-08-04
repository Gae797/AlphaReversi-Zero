import tensorflow as tf

class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate, initial_current_step, decreasing_steps):

        self.learning_rate = initial_learning_rate
        self.current_step = initial_current_step
        self.decreasing_steps = decreasing_steps

    def __call__(self, step):

        self.current_step += 1

        for decreasing_step in self.decreasing_steps:
            if self.current_step==decreasing_step:
                self.learning_rate = self.learning_rate / 10.0

        return self.learning_rate
