'''
This module contains an interface for all the agents to play
'''

from abc import ABC, abstractmethod

class AgentInterface(ABC):

    @abstractmethod
    def play(self, board):
        pass

    @property
    @abstractmethod
    def is_external_engine(self):
        pass
