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
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
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
    # we can save the model and reload it at anytime in the future
    model.save('imbd_RNN.h5') 
    results = model.evaluate(test_data, test_labels)
    print(results)
    plot_history(history)


def encode_text(text,word_index):
    '''
        Since our reviews are encoded well need to convert any review that we write into
        that form so the network can understand it. 
        To do that well load the encodings from the dataset and use them to encode our own data.
    '''
    tokens = keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return sequence.pad_sequences([tokens], MAXLEN)[0]

# while were at it lets make a decode function
def decode_integers(integers,reverse_word_index):
    PAD = 0
    text = ""
    for num in integers:
      if num != PAD:
        text += reverse_word_index[num] + " "
    return text[:-1]
  

def test_predict():
    word_index = imdb.get_word_index()
    text = "that movie was just amazing, so amazing"
    encoded = encode_text(text,word_index)
    print(encoded)
    reverse_word_index = {value: key for (key, value) in word_index.items()}
    print(decode_integers(encoded,reverse_word_index))


# now time to make a prediction

def encode_decode(model,text):
    word_index = imdb.get_word_index()
    encoded_text = encode_text(text,word_index)
    pred = np.zeros((1,250))
    pred[0] = encoded_text
    result = model.predict(pred) 
    print(result[0])


def predict():
    # If prediction < 0.5 then the prediction is positive.
    model = tf.keras.models.load_model('imbd_RNN.h5')
    positive_review = "That movie was! really loved it and would great watch it again because it was amazingly great"
    encode_decode(model,positive_review)

    negative_review = "that movie really sucked but amazing.  I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
    encode_decode(model,negative_review)


def run():
    TRAIN = False

    if TRAIN:
        RNN()
    else:
        test_predict()
        predict()




if __name__ == '__main__':
    run()