# The example function below keeps track of the opponent's history and 
# lays whatever the opponent played two plays ago. It is not a very good 
# player so you will need to change the code to pass the challenge.

import predictor
import constants

markov_matrix = predictor.MarkovPredictor(0.95,1)
ocurrences =  {} # Key : pattern, Value : appearences

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


def update_occurrences(pattern):
    global ocurrences
    if pattern in ocurrences:
        ocurrences[pattern] += 1
    else:
        ocurrences[pattern] = 1


def player(prev_play, opponent_history=[]):
    global ocurrences
    if prev_play != '':
        opponent_history.append(prev_play)

    #bots plays with the background. Heuristic game with record could beat them if find a pattern.
    #return player_markov(opponent_history)

    #Size to explore back. Size 4 and 5 beat all but sometimes fail vs Abbey. 
    #W/R vs Abbey = 58/65 % prev_n = 5
    prev_n = 5

    if len(opponent_history) > prev_n:
        #Create string with the pattern.
        find = ''.join(opponent_history[-(prev_n):])
        #Update number pattern x appear.
        update_occurrences(find)
        #Check possible combinations.
        max = 0
        guess = ''
        for k in constants.keys:
            #One previous to predict.
            choosen = ''.join(opponent_history[(-prev_n+1):]) + k
            #Check if pattern exists and get the most used.
            if not choosen in ocurrences:
                ocurrences[choosen] = 0
            if (ocurrences[choosen] > max):
                guess = k
                max = ocurrences[choosen]
        if guess != '':
            return constants.beat_dict[guess]

    return predictor.RandomPredictor().predict()
