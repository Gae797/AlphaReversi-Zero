'''
This module starts the training of the model
'''

from src.rl.training.local_trainer import Trainer

if __name__ == '__main__':

    trainer = Trainer()
    trainer.run()
