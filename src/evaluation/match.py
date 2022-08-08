from src.agents.agent_interface import AgentInterface
from src.game import Game

class Match:

    def __init__(self, agent_1, agent_2, match_length, time_per_move=5, use_gui=True):

        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.match_length = match_length
        self.time_per_move = time_per_move
        self.use_gui = use_gui

        self.scores_agent_1 = 0.0
        self.scores_agent_2 = 0.0

    def play(self):

        for i in range(self.match_length):

            print("Playing game number {}".format(i+1))

            if i%2==0:
                game = Game(self.agent_1, self.agent_2, self.time_per_move, self.use_gui, False, True)
                result = game.play_game()
                self.scores_agent_1 += result[0]
                self.scores_agent_2 += result[1]
            else:
                game = Game(self.agent_2, self.agent_1, self.time_per_move, self.use_gui, False, True)
                result = game.play_game()
                self.scores_agent_1 += result[1]
                self.scores_agent_2 += result[0]

        print("Match completed")
        print("{}    {}-{}    {}".format(self.agent_1.name,
                                        self.scores_agent_1,
                                        self.scores_agent_2,
                                        self.agent_2.name))

        return (self.scores_agent_1, self.scores_agent_2)
