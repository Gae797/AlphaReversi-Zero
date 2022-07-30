from abc import ABC, abstractmethod

class AgentInterface(ABC):

    @abstractmethod
    def play(self, board, time):
        pass
