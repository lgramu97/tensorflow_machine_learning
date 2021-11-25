
from random import random
import predictor
import constants

def run():

    random_predictor = predictor.RandomPredictor()

    markov_matrix = predictor.MarkovPredictor()
    markov_matrix.print_matrix()

    for _ in range(0,15):
        print(markov_matrix.play('RR'))
        markov_matrix.update_mtrx('RR',random_predictor.predict())
    markov_matrix.print_matrix()
    

    beats_dict = constants.beats_dict

    x = random_predictor.predict()
    print('{} beats {}'.format(x,beats_dict[x]))

if __name__ == '__main__':
    run()