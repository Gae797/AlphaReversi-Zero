import numpy as np
import matplotlib.pyplot as plt
import os

from src.evaluation.match import Match
from src.agents.alpha_reversi_agent import AlphaReversiAgent
from src.agents.edax_agent import EdaxAgent

from src.environment.config import *
from src.rl.config import *

class Evaluator:

    def __init__(self, weights, network_depth, engine_depth, match_length, use_gui=False):

        self.network_moves_evals = []
        self.engine_moves_evals = []

        self.largest_drops = []

        self.alpha_reversi_agent = AlphaReversiAgent(BOARD_SIZE, N_RESIDUAL_BLOCKS, weights, network_depth)
        self.edax_agent = EdaxAgent(engine_depth)

        self.match = Match(self.alpha_reversi_agent, self.edax_agent, match_length,
                        use_gui=use_gui, start_from_random_position=True)

        self.alpha_reversi_agent.attach_evaluator(self)
        self.match.attach_evaluator(self)

    def get_evaluations(self, network_move_eval, engine_move_eval):

        self.network_moves_evals.append(network_move_eval)
        self.engine_moves_evals.append(engine_move_eval)

    def notify_game_end(self):

        self.engine_moves_evals = self.engine_moves_evals[1:]

        if len(self.network_moves_evals)==len(self.engine_moves_evals)+1:
            self.network_moves_evals = self.network_moves_evals[:-1]

        assert len(self.network_moves_evals)==len(self.engine_moves_evals)

        differences = [abs(network_eval-engine_eval)
                    for network_eval,engine_eval in zip(self.network_moves_evals, self.engine_moves_evals)]

        self.largest_drops.append(max(differences))

        self.engine_moves_evals.clear()
        self.network_moves_evals.clear()

    def evaluate(self):

        print("Evaluating model...")

        results = self.match.play()

        score = results[0]
        avg_value_drop = np.mean(self.largest_drops)

        return score, avg_value_drop


def evaluate_generations(generations, network_depth, engine_depth, match_length, use_gui=False):

    avg_value_drops = []
    scores = []

    for generation in generations:

        weights = os.path.join(WEIGHTS_PATH, "Generation {}".format(generation), "variables")

        evaluator = Evaluator(weights, network_depth, engine_depth, match_length, use_gui)
        score, avg_value_drop = evaluator.evaluate()
        scores.append(score)
        avg_value_drops.append(avg_value_drop)

    x_points = np.array(generations)
    y_scores = np.array(scores)
    y_drops = np.array(avg_value_drops)

    plt.plot(x_points, y_scores, label="Score against Edax")
    plt.plot(x_points, y_drops, label="Average drop value")
    plt.legend()
    plt.show()
