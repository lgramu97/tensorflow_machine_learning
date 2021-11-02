'''
The problem we will consider here is classifying 10 different everyday objects.
 The dataset we will use is built into tensorflow and called the [**CIFAR Image Dataset.**]
 (https://www.cs.toronto.edu/~kriz/cifar.html) 
 It contains 60,000 32x32 color images with 6000 images of each class. 

The labels in this dataset are the following:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck
'''

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


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


def plot_image(image,name):
    plt.figure()
    plt.imshow(image)
    plt.title(name)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def spy_data(train_images,train_labels,class_names):
    #Shape data is: (50000,32,32,3) 500000imgs of 28 x 28 x 3 -> numpy array
    print('Data shape: ', train_images.shape)
    print('Labels shape: ', train_labels.shape) # 50000 img x 1
    print('Lets see one pixel value: ', train_images[0,23,23])
    print('Pixel values: ',np.unique(train_images[0])) #Every pixel between 0-255. Color Images.
    print('Labels: ', train_labels[:10,0])  # let's have a look at the first 10 training labels)
    labels = np.unique(train_labels)
    print('Labels values: ', labels, 'Total classes: ', len(labels))

    #Show images
    i = 7
    plot_image(train_images[i],str(class_names[train_labels[i,0]] + ' = ' + str(train_labels[i,0])))


def data_preprocessing(train_images,test_images):
    '''
        Normalize values between 0-1
    '''
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return train_images, test_images


def get_model():
    #Pytorch [C,H,W]. Tensorflow/Keras (H,W,C)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    #Dense layer for classificate.
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    return model


def run():
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
    #  LOAD AND SPLIT DATASET
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    #Lets check the data.
    if False:
        spy_data(train_images,train_labels,class_names)

    # Preprocessing: Normalize pixel values to be between 0 and 1
    train_images, test_images = data_preprocessing(train_images,test_images)

    model = get_model()
    #Check model architecture.
    model.summary()

    #First compile the model with Optimizer, Loss Function and Metrics.
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=4, 
                        validation_data=(test_images, test_labels))

    plot_history(history)


if __name__ == '__main__':
    run()