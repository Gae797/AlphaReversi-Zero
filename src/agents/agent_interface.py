from abc import ABC, abstractmethod

class AgentInterface(ABC):

    @abstractmethod
    def play(self, board, timer):
        pass

    @property
    @abstractmethod
    def is_external_engine(self):
        pass
