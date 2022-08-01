import numpy as np

from src.environment.board import Board

class Node:

    def __init__(self, board, parent=None):

        self.board = board

        self.estimated_value = self.board.reward
        self.estimated_policy = None

        self.visit_count = 0
        self.total_outcome = 0.0
        self.average_outcome = 0.0

        self.children = None #actions
        self.parent = parent

    def set_estimation(self, prediction):

        global_policy = prediction[0]
        self.estimated_policy = np.where(global_policy>0)[0].tolist()

        self.estimated_value = prediction[1]

    def expand(self):

        self.children = []

        legal_moves = board.legal_moves["indices"]

        for action in legal_moves:
            new_board = self.board.move(action)
            child = Node(new_board, self)
            children.append((child,action))

    def backup_update(self, value):

        self.visit_count += 1
        self.total_outcome += value
        self.average_outcome = self.total_outcome / self.visit_count

        if self.parent is not None:
            self.parent.backup_update(value)

    def is_evaluated(self):

        return self.estimated_value is not None

    def is_leaf(self):

        return self.visit_count==0

    def is_expanded(self):

        return self.children is not None
