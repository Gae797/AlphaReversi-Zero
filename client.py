'''
This module is used to run a remote trainer
'''

from src.rl.training.remote_trainer import RemoteTrainer

if __name__ == '__main__':

    remote_trainer = RemoteTrainer()
    remote_trainer.run()
