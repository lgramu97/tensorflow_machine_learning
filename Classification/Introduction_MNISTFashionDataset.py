# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


'''
MNIST Fashion Dataset. This is a dataset that is included in keras.

This dataset includes 60,000 images for training and 10,000 images for validation/testing.
'''

def plot_image(image,name):
    plt.figure()
    plt.imshow(image)
    plt.title(name)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def spy_data(train_images,train_labels,class_names):
    #Shape data is: (60000,28,28) 600000imgs of 28 x 28 -> numpy array
    print('Data shape: ', train_images.shape)
    print('Labels shape: ', train_labels.shape)
    print('Lets see one pixel value: ', train_images[0,23,23])
    print('Pixel values: ',np.unique(train_images[0])) #Every pixel between 0-255. Gray Scale Images.
    print('Labels: ', train_labels[:10])  # let's have a look at the first 10 training labels)
    labels = np.unique(train_labels)
    print('Labels values: ', labels, 'Total classes: ', len(labels))

    #Show images
    i = 2
    plot_image(train_images[i],str(class_names[train_labels[i]] + ' = ' + str(train_labels[i])))


def data_preprocessing(train_images,test_images):
    '''
        Normalize values between 0-1
    '''
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return train_images, test_images


def buid_model(input_shape,classes):
    '''
        **Layer 1:** This is our input layer and it will conist of 784 neurons. 
        We use the flatten layer with an input shape of (28,28) to denote that our input should
        come in in that shape. The flatten means that our layer will reshape the shape (28,28)
        array into a vector of 784 neurons so that each pixel will be associated with one neuron.

        **Layer 2:** This is our first and only hidden layer. The *dense* denotes that
        this layer will be fully connected and each neuron from the previous layer connects to
        each neuron of this layer. It has 128 neurons and uses the rectify linear unit activation
        function.

        **Layer 3:** This is our output later and is also a dense layer.
        It has 10 neurons that we will look at to determine our models output.
        Each neuron represnts the probabillity of a given image being one of the 10 different classes.
        The activation function *softmax* is used on this layer to calculate a probabillity distribution
        for each class. This means the value of any neuron in this layer will be between 0 and 1,
        where 1 represents a high probabillity of the image being that class.
    '''
    model = keras.Sequential([
            keras.layers.Flatten(input_shape=input_shape),  # input layer (1)
            keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
            keras.layers.Dense(classes, activation='softmax') # output layer (3)
        ])
    return model


def train_model(model,train_images,train_labels):
    # we pass the data, labels and epochs and watch the magic!
    model.fit(train_images, train_labels, epochs=10)  


def test_model(model,test_images,test_labels):
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 
    print('Test accuracy:', test_acc)
    print('Test loss:', test_loss)


def predict_images(model,test_images,class_names):
    predictions = model.predict(test_images)
    i = 0
    print(class_names[np.argmax(predictions[i])])
    plot_image(test_images[0],class_names[np.argmax(predictions[i])])


''' Script made by freecodecamp'''
COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]

  show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  plt.title("Excpected: " + label)
  plt.xlabel("Guess: " + guess)
  plt.colorbar()
  plt.grid(False)
  plt.show()


def get_number():
  while True:
    num = input("Pick a number: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
    else:
      print("Try again...")
''' finish script made by freecodecamp'''


def run():
    # load dataset
    fashion_mnist = keras.datasets.fashion_mnist  
    # split into tetsing and training
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    #Show data information
    if False:
        spy_data(train_images,train_labels,class_names)
    #Pre-process data. (Normalization)
    train_images, test_images = data_preprocessing(train_images,test_images)
    print(train_images.shape)
    #Build the model
    model = buid_model(input_shape=(28,28),classes=10)
    #Compile model with Optimizer, Loss function and Metrics evaluation.
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    #Train model
    train_model(model,train_images,train_labels)
    #Test model
    test_model(model,test_images,test_labels)
    #Predict images
    predict_images(model,test_images,class_names)

    #Using script from freecodecamp.
    while input('Continue? (Y/N)').upper() == "Y":
        num = get_number()
        image = test_images[num]
        label = test_labels[num]
        predict(model, image, label)


if __name__ == '__main__':
    run()