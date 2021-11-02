from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output

import tensorflow as tf
import os

'''
Lets try Titanic data
'''

def load_datasets(path):
    '''
        Try load it locally, else download from internet.
    '''
    try:
        train_path = os.path.join(path,'train.csv')
        eval_path = os.path.join(path,'eval.csv')
        df_train = pd.read_csv(train_path) 
        df_eval = pd.read_csv(eval_path)
    except:
        df_train = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
        df_eval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
    return df_train,df_eval

def plot_data_information(x_train,y_train):
    '''
        Plot some data information to know what kind of data we are manipulating.
    '''
    #Lets see the age distribution
    x_train.age.hist(bins=20)
    plt.show()
    #Lets see the sex distribution
    x_train.sex.value_counts().plot(kind='barh')
    plt.show()
    #Lets see the class distribution
    x_train['class'].value_counts().plot(kind='barh')
    plt.show()

    #% survived by sex.
    pd.concat([x_train, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
    plt.show()


def convert_categorical_data(x_train):
    '''
        Encode categorical data and convert it to numeric data.
    '''

    CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
    NUMERIC_COLUMNS = ['age', 'fare']
    
    print(x_train['embark_town'].unique())
    
    feature_columns = []
    for feature_name in CATEGORICAL_COLUMNS:
        # gets a list of all unique values from given feature column
        vocabulary = x_train[feature_name].unique() 
        feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

    for feature_name in NUMERIC_COLUMNS:
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
    #feature_columns is a list of list inside (key,values,type,default_value,num_oov_buckets)
    print(feature_columns)
    return feature_columns


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    ''' 
    Tensorflow model needs a dataset object
    '''
    def input_function():  # inner function, this will be returned
        # create tf.data.Dataset object with data and its label
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  
        if shuffle:
            ds = ds.shuffle(1000)  # randomize order of data
        # split dataset into batches of 32 and repeat process for number of epochs
        ds = ds.batch(batch_size).repeat(num_epochs)  
        return ds  # return a batch of the dataset
    return input_function  # return a function object for use
   

if __name__ == '__main__':
    dataset_path = '/home/lautaro/Cursos/FreeCodeCampMachineLearning/'
    df_train, df_eval = load_datasets(dataset_path)
    #Get labels (Survived 1 - No Survived 0)
    y_train = df_train.pop('survived')
    y_eval = df_eval.pop('survived')
    #print(df_train.loc[0], ' Survived = ', y_train.loc[0])
    

    if False:
        print('Train shape: ' ,df_train.shape)
        print('Eval shape: ', df_eval.shape)
        print('Spy on the data: \n' , df_train.head())
        print('More details: \n', df_train.describe())
        plot_data_information(df_train,y_train)
        '''
        After analyzing this information, we should notice the following:
        - Most passengers are in their 20's or 30's 
        - Most passengers are male
        - Most passengers are in "Third" class
        - Females have a much higher chance of survival
        '''
    
    #Pre-proccess data.
    feature_cols = convert_categorical_data(df_train)

    #call the input_function to get a dataset object to feed to the model
    train_input_fn = make_input_fn(df_train, y_train)  
    eval_input_fn = make_input_fn(df_eval, y_eval, num_epochs=1, shuffle=False)

    #Create linear estimator
    #We create a linear estimtor by passing the feature columns we created earlier
    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_cols)

    #Train needs an input function, so we pass the function we made.
    linear_est.train(train_input_fn)  # train
    #get model metrics/stats by testing on tetsing data
    result = linear_est.evaluate(eval_input_fn)  
    #the result variable is simply a dict of stats about our model
    print(result['accuracy'])
    #Print all metrics.
    print(result)

    #We want to do a prediction to get the probabilitie of survived.
    results = list(linear_est.predict(eval_input_fn))
    #Print some person information.
    person_idx = 1
    print(df_eval.iloc[person_idx])
    #check person survived or not.
    print('Survived? : ' , y_eval[person_idx])
    #Index the predictions and get the probabilities of survive. 0 not survived. 1 survived.
    print('Probabilitie survived: ',results[person_idx]['probabilities'][1])

    #Show probabilities in a histogram.
    probs = pd.Series([pred['probabilities'][1] for pred in results])
    probs.plot(kind='hist', bins=20, title='predicted probabilities')
    plt.show()

