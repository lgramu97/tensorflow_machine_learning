# The example function below keeps track of the opponent's history and 
# lays whatever the opponent played two plays ago. It is not a very good 
# player so you will need to change the code to pass the challenge.

import predictor
import constants

markov_matrix = predictor.MarkovPredictor(0.95,1)

def use_matrix(pair):
    global markov_matrix
    guess = markov_matrix.play(pair)
    markov_matrix.update_mtrx(pair,constants.beat_dict[guess])
    return constants.beat_dict[guess]


def player_markov(opponent_history):
    if len(opponent_history) > 2:
        if '' in opponent_history[-2:]:
            return predictor.RandomPredictor.predict()
        pair = opponent_history[-2]+opponent_history[-1]
        return use_matrix(pair)
    return predictor.RandomPredictor.predict()


def player(prev_play, opponent_history=[]):
    opponent_history.append(prev_play)

    return player_markov(opponent_history)


     
