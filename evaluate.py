from src.environment.config import *
from src.rl.config import *

from src.evaluation.match import Match
from src.agents.alpha_reversi_agent import AlphaReversiAgent
from src.agents.random_agent import RandomAgent
from src.agents.human_agent import HumanAgent

weights_1 = os.path.join(WEIGHTS_PATH, "Generation {}".format(34), "variables")
weights_2 = os.path.join(WEIGHTS_PATH, "Generation {}".format(5), "variables")

agent_1 = AlphaReversiAgent(BOARD_SIZE, N_RESIDUAL_BLOCKS, weights_1, 200, name="Generation 8")
agent_2 = AlphaReversiAgent(BOARD_SIZE, N_RESIDUAL_BLOCKS, weights_2, 200, name="Generation 5")
agent_rnd = RandomAgent()
agent_human = HumanAgent()

match = Match(agent_1, agent_rnd, 10)
match.play()
