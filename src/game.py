from src.agents.agent_interface import AgentInterface
from src.environment.board import Board
from src.gui.window import Window

class Game:

    def __init__(self, white_agent, black_agent, time_per_move=5, use_gui=True, draw_legal_moves=False, show_names=False):

        self.white_agent = white_agent
        self.black_agent = black_agent
        self.time_per_move = time_per_move
        self.use_gui = use_gui
        self.show_names = show_names

        self.current_board = Board()

        #TODO GUI in other thread?
        if use_gui:
            self.game_window = Window(draw_legal_moves)

    def play_game(self):

        if self.white_agent.is_external_engine:
            self.white_agent.start_new_game()
        if self.black_agent.is_external_engine:
            self.black_agent.start_new_game()

        result = self.play_move()

        if self.white_agent.is_external_engine:
            self.white_agent.close_game()
        if self.black_agent.is_external_engine:
            self.black_agent.close_game()

        return result

    def play_move(self):

        turn = self.current_board.turn
        playing_agent = self.white_agent if turn else self.black_agent
        opponent_agent = self.white_agent if not turn else self.black_agent

        move_number = playing_agent.play(self.current_board, self.time_per_move)
        self.current_board = self.current_board.move(move_number)

        if self.use_gui:
            self.game_window.update(self.current_board)

        if opponent_agent.is_external_engine:
            opponent_agent.update_position(move_number)
            if turn==self.current_board.turn:
                opponent_agent.force_pass()

        if playing_agent.is_external_engine and turn==self.current_board.turn:
            playing_agent.force_pass()

        if self.current_board.is_terminal:
            if self.current_board.reward==0:
                print("Draw")
                result = (0.5, 0.5)
            elif self.current_board.reward==1:
                result = (1.0, 0.0)
                if self.show_names:
                    print("{} won".format(self.white_agent.name))
                else:
                    print("White won")
            else:
                result = (0.0, 1.0)
                if self.show_names:
                    print("{} won".format(self.black_agent.name))
                else:
                    print("Black won")

            return result

        else:
            return self.play_move()
