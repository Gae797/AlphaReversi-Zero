from src.agents.agent_interface import AgentInterface
from src.environment.board import Board
from src.gui.window import Window

class Game:

    def __init__(self, white_agent, black_agent, time_per_move=5, use_gui=True, draw_legal_moves=False):

        self.white_agent = white_agent
        self.black_agent = black_agent
        self.time_per_move = time_per_move
        self.use_gui = use_gui

        self.current_board = Board()

        #TODO GUI in other thread
        if use_gui:
            self.game_window = Window(draw_legal_moves)

    def play(self):

        turn = self.current_board.turn
        playing_agent = self.white_agent if turn else self.black_agent

        move_number = playing_agent.play(self.current_board, self.time_per_move)
        self.current_board = self.current_board.move(move_number)
        if self.use_gui:
            self.game_window.update(self.current_board)

        if self.current_board.is_terminal:
            if self.current_board.reward==0:
                print("Draw")
            elif self.current_board.reward==1:
                print("White won")
            else:
                print("Black won")

        else:
            self.play()
