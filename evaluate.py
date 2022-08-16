from src.environment.config import *
from src.rl.config import *

from src.evaluation.match import Match
from src.agents.alpha_reversi_agent import AlphaReversiAgent
from src.agents.random_agent import RandomAgent

weights_1 = os.path.join(WEIGHTS_PATH, "Generation {}".format(36), "variables")
weights_2 = os.path.join(WEIGHTS_PATH, "Generation {}".format(10), "variables")

agent_1 = AlphaReversiAgent(BOARD_SIZE, N_RESIDUAL_BLOCKS, weights_1, 400, name="Generation 36")
agent_2 = AlphaReversiAgent(BOARD_SIZE, N_RESIDUAL_BLOCKS, weights_2, 400, name="Generation 10")

match = Match(agent_1, agent_2, 10)
match.play()
