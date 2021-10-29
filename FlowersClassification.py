'''
Where regression was used to predict a numeric value, classification is used to seperate data
points into classes of different labels. In this example we will use a TensorFlow estimator to 
classify flowers.
'''

from __future__ import absolute_import, division, print_function, unicode_literals
from warnings import catch_warnings

import tensorflow as tf
import pandas as pd
import os

from tensorflow._api.v2 import feature_column

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

def load_datasets(path):
    '''
        Load Flower Iris dataset.
    '''
    # Lets define some constants to help us later on  
    train_path = os.path.join(path,'iris_training.csv')
    test_path =  os.path.join(path,'iris_test.csv')
    if not os.path.isfile(train_path):
        #Use keras to grab dataset and read them into pandas dataframe.
        train_path = tf.keras.utils.get_file(   
        "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
        test_path = tf.keras.utils.get_file(
        "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")
    
    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    return train, test
        
def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)

def get_features_columns(dataframe):
    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in dataframe.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    return my_feature_columns

def predict_input(features, batch_size=256):
   # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

def predict(model):
    '''
        Use the model trainned to predict new inputs.
    '''
    features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
    predict = {}
    print("Please type numeric values as prompted.")
    #For all the features we need a numeric value. (Manual)
    if False:
        for feature in features:
            valid = True
            while valid: 
                val = input(feature + ": ")
                try:
                    val = float(val)
                    valid = False
                except:
                    pass
            predict[feature] = [float(val)]

    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
        }
    predictions = classifier.predict(input_fn=lambda: predict_input(predict_x))
    print('\n Predictions: ',predictions)
    i = 0
    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{}" ({:.1f}%) and expected is "{}"'
            .format(SPECIES[class_id], 100 * probability,expected[i]))
        i += 1

if __name__ == '__main__':

    #3 classes flowers: Setosa-Versicolor-Virginica
    dataset_path = '/home/lautaro/Cursos/FreeCodeCampMachineLearning/'
    x_train, x_test = load_datasets(dataset_path)

    #Looks like the last column is the label.
    print(x_train.head())

    #Extract the labels.
    y_train = x_train.pop('Species')
    y_test = x_test.pop('Species')

    #Look again.
    print(x_train.head())
    print(y_train.head())

    #Shape
    print('Shape x train:', x_train.shape) #(120,4)
    print('Shape x test:', x_test.shape) #(30,4)

    #Get all data (numerical)
    feature_column = get_features_columns(x_train)
    print(feature_column, '\n' , "len must be 4 and its = " , len(feature_column))

    #Build model with tensorflow.estimator (lot of pre charged models.)
    #Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
    classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_column,
    hidden_units=[30, 10],# Two hidden layers of 30 and 10 nodes respectively.
    n_classes=3)# The model must choose between 3 classes.

    x = lambda : print("Hello World")
    x()

    #Train the model
    classifier.train(input_fn=lambda: input_fn(x_train, y_train, training=True),steps=5000)
    # We include a lambda to avoid creating an inner function previously

    #Eval the model
    eval_result = classifier.evaluate(input_fn=lambda: input_fn(x_test, y_test, training=False))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    print('\nAll metrics: ', eval_result)
    
    #Predict some data input.
    predict(classifier)

