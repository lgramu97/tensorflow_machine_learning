
import random

from numpy.core.defchararray import index
import constants
import numpy as np
import itertools

class RandomPredictor():

    @staticmethod
    def predict():
        return random.choice(constants.keys)


class MarkovPredictor():

    def __init__(self,alpha,N) -> None:
        self.alpha = alpha
        self.order = N
        self.markov_matrix = self.init_matrix()
        

    def print_matrix(self):
        for row in self.markov_matrix.keys():
            print(row)
            for k in self.markov_matrix[row]:
                print('--->',k , '  ', self.markov_matrix[row][k])
    

    def init_rows(self):
                          
        rows = ['R', 'P', 'S']

        for i in range((self.order * 2 - 1)):
            key_len = len(rows)
            for i in itertools.product(rows, ''.join(rows)):
                rows.append(''.join(i))
            rows = rows[key_len:]
    
        return rows


    def init_matrix(self):
        matrix = {}

        rows = self.init_rows()
        for row in rows:
            matrix[row] = { 'R': { 'prob' : 1/3, 'total' : 0},
                            'P': { 'prob' : 1/3, 'total' : 0},
                            'S': { 'prob' : 1/3, 'total' : 0}}
        return matrix


    def update_mtrx(self,row,key):

        for i in self.markov_matrix[row]:
            self.markov_matrix[row][i]['total'] = self.alpha * self.markov_matrix[row][i]['total']

        #Last game pair + 1
        self.markov_matrix[row][key]['total'] = self.markov_matrix[row][key]['total'] + 1

        total = 0
        for i in self.markov_matrix[row]:
            total += self.markov_matrix[row][i]['total']

        
        #Update all probs.
        for k in self.markov_matrix[row]:
            self.markov_matrix[row][k]['prob'] = (self.markov_matrix[row][k]['prob']/total)
        

    def probs_array(self,row):
        out = []
        for k in self.markov_matrix[row]:
            out.append(self.markov_matrix[row][k]['prob'])
        return out


    def play(self,row):
        probs = self.probs_array(row)
        if max(probs) == min(probs):
            return RandomPredictor.predict()
        index_max = np.argmax(probs)
        return constants.keys[index_max]


                
        