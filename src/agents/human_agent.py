'''
This agent is controlled by human textual inputs
'''

from src.agents.agent_interface import AgentInterface
import src.environment.coordinate_handler as coordinate_handler

from threading import Thread

class HumanAgent(AgentInterface):

    def __init__(self, name="Human agent"):

        self.name = name

    def play(self, board):

        #Create a separated thread to ask for a move

        thread_result = []
        thread = Thread(target=self.play_thread,args=(board, thread_result))

        thread.start()
        thread.join()

        move = thread_result[0]

        return move

    def play_thread(self, board, thread_result):

        #Print legal moves
        legal_moves = board.legal_moves["indices"]

        coordinates = [coordinate_handler.move_to_coordinate(move) for move in legal_moves]
        print(coordinates)

        #Ask the human agent to select a move
        coordinate = input("Choose a move: ")
        coordinate = coordinate.lower()

        if not coordinate in coordinates:
            raise "Invalid move!"

        result = coordinate_handler.coordinate_to_move(coordinate)

        thread_result.append(result)

    @property
    def is_external_engine(self):
        return False
