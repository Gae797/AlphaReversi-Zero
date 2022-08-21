'''
Match class is used to start a match of specified length against two agents
'''

from src.agents.agent_interface import AgentInterface
from src.environment.game import Game

class Match:

    def __init__(self, agent_1, agent_2, match_length,
                use_gui=True, start_from_random_position=False):

        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.match_length = match_length
        self.use_gui = use_gui
        self.start_from_random_position = start_from_random_position

        self.scores_agent_1 = 0.0
        self.scores_agent_2 = 0.0

        self.evaluator = None

    def play(self):

        for i in range(self.match_length):

            print("Playing game number {}".format(i+1))

            if i%2==0:
                game = Game(self.agent_1,
                            self.agent_2,
                            self.use_gui,
                            False,
                            True,
                            self.start_from_random_position)

                result = game.play_game()
                self.scores_agent_1 += result[0]
                self.scores_agent_2 += result[1]

            else:
                game = Game(self.agent_2,
                            self.agent_1,
                            self.use_gui,
                            False,
                            True,
                            self.start_from_random_position)

                result = game.play_game()
                self.scores_agent_1 += result[1]
                self.scores_agent_2 += result[0]

            if self.evaluator is not None:
                self.evaluator.notify_game_end()

        print("Match completed")
        print("{}    {}-{}    {}".format(self.agent_1.name,
                                        self.scores_agent_1,
                                        self.scores_agent_2,
                                        self.agent_2.name))

        return (self.scores_agent_1, self.scores_agent_2)

    def attach_evaluator(self, evaluator):

        self.evaluator = evaluator

    def remove_evaluator(self):

        self.evaluator = None
