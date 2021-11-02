from PIL.Image import init
import tensorflow
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input ,Dense, MaxPool2D, BatchNormalization, Flatten, GlobalAvgPool2D
from tensorflow.python.keras.engine.input_layer import InputLayer

from deeplearning_models import functional_model, MyCustomModel, MyCustomModel2

# tensorflow.keras.Sequential : forma mas facil de construir un modelo.
seq_model = tensorflow.keras.Sequential(
    [
        Input(shape=(28,28,1)), #gray scale images 28x28
        Conv2D(filters=32,kernel_size=(3,3),activation='relu'), #32 filters of 3x3 and then Relu.
        Conv2D(64,(3,3),activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        Conv2D(128,(3,3),activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        Conv2D(256,(3,3),activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        GlobalAvgPool2D(),
        Dense(64, activation='relu'),
        Dense(10,activation='softmax') #Output layer, Classification layer.
    ]
)


if __name__== '__main__':
    
    #Load mnist dataset.
    (x_train,y_train), (x_test,y_test) = tensorflow.keras.datasets.mnist.load_data()

    #Check data shape and size
    print("x_train.shape = ", x_train.shape)
    print("y_train.shape = ", y_train.shape)
    print("x_test.shape = ", x_test.shape)
    print("y_test.shape = ", y_test.shape)

    if False:
        display_some_examples(x_train,y_train)

    #Normalize data. Gradient move faster to global minimum
    x_train = x_train.astype('float32') / 255 #8bits data type -> float 
    x_test = x_test.astype('float32') / 255 

    x_train = np.expand_dims(x_train,axis=-1) # h , w , n
    x_test = np.expand_dims(x_test,axis=-1) # h , w , n

    #model = functional_model( )
    model = MyCustomModel2()
    #Compile model. https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

    # categorical corrs entropy need one-hot-encode
    # label: 2
    # one hot: [0,0,1,0,0,0,0,0,0,0,0,0]
    # Transform labels: y_train = tensorflow.keras.utils.to_categorical(y_train, 10)

    '''
    compile(
    optimizer='rmsprop', loss=None, metrics=None, loss_weights=None,
    weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs)

    or

    model.compile(optimizer=tf.keras.optimizer.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.FalseNegatives()])
    '''

    #Fit model. https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
    model.fit(x_train,y_train,batch_size=64, epochs=3, validation_split=0.2) #80% train for training, 20% validation

    #Evaluate model. (test set) https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate
    model.evaluate(x_test,y_test,batch_size=64)


