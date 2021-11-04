'''
MOVIE REVIEW DATASET.
Well start by loading in the IMDB movie review dataset from keras. 
This dataset contains 25,000 reviews from IMDB where each one is already preprocessed and
has a label as either positive or negative. Each review is encoded by integers that represents
how common a word is in the entire dataset. For example, a word encoded by the
integer 3 means that it is the 3rd most common word in the dataset.
'''

from keras.datasets import imdb
from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

VOCAB_SIZE = 88584
MAXLEN = 250
BATCH_SIZE = 64


def plot_history(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def spy_on_data(train_data,train_labels,test_data,test_labels):
    print('Train data: ', 'Type_', type(train_data[0]), ' Total data_', len(train_data))
    print('Train data 0' , train_data[0], len(train_data[0]))
    print('Train data 1' , train_data[1], len(train_data[1]))

    print('Train labels: ', 'Type_', type(train_labels[0]), ' Total data_', len(train_labels))
    print('Train labels 0' , train_labels[0])

    print('Test data: ', 'Type_', type(test_data[0]), ' Total data_', len(test_data))
    print('Test data 0' , test_data[0])
    print('Test labels: ', 'Type_', type(test_labels[0]), ' Total data_', len(test_labels))
    print('Test labels 0' , test_labels[0])


def preprocess_data(train_data,test_data):
    '''
    - if the review is greater than 250 words then trim off the extra words
    - if the review is less than 250 words add the necessary amount of 0's to make it equal to 250.
    '''
    train_data = sequence.pad_sequences(train_data, MAXLEN)
    test_data = sequence.pad_sequences(test_data, MAXLEN)
    return train_data, test_data


def define_model():
    '''
        Positive review: prob > 0.5 = 1
        Negative review: prob < 0.5 = 0
    '''
    model = tf.keras.Sequential([
            tf.keras.layers.Embedding(VOCAB_SIZE, 32),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
    model.summary()
    return model


def RNN():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)
    if False:
        spy_on_data(train_data,train_labels,test_data,test_labels)
    #Preprocess data
    train_data, test_data = preprocess_data(train_data,test_data)
    #Define the RNN network
    model = define_model()
    #Train out model
    model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=['acc'])
    history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)
    #Plot the results
    plot_history(history)


def run():
    RNN()


if __name__ == '__main__':
    run()