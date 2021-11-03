
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
keras = tf.keras


def spy_images_from_raw(train_image,train_label):
    # display 2 images from the dataset
    for image, label in train_image.take(5):
        plt.figure()
        plt.imshow(image)
        plt.title(train_label(label))
    plt.show()


def load_dataset():
    #*cats_vs_dogs* dataset from the modoule tensorflow_datatsets
    #(image, label) pairs where images have different dimensions and 3 color channels
    (raw_train, raw_validation, raw_test), metadata = tfds.load(
            'cats_vs_dogs',
            split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
            with_info=True,
            as_supervised=True,
            )
    #split the data manually into 80% training, 10% testing, 10% validation
    #creates a function object that we can use to get labels
    get_label_name = metadata.features['label'].int2str  
    
    return raw_train, raw_validation, raw_test, get_label_name


def format_example(image, label):
  """
  returns an image that is reshaped to IMG_SIZE
  """
  IMG_SIZE = 160 # All images will be resized to 160x160
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label


def spy_data(train):
    for image, label in train.take(5):
        plt.figure()
        plt.imshow(image)
        plt.title(get_label_name(label))
    plt.show()


def preprocess_data(raw_train,raw_validation,raw_test):
    #map take all of the elements and apply the function.
    train = raw_train.map(format_example)
    validation = raw_validation.map(format_example)
    test = raw_test.map(format_example)

    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 1000

    train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    validation_batches = validation.batch(BATCH_SIZE)
    test_batches = test.batch(BATCH_SIZE)

    return train_batches, validation_batches, test_batches


def run():
    #Download and get partitions. get_label_name is a method, recive a number and get the label.
    raw_train, raw_validation, raw_test, get_label_name = load_dataset()
    
    #Spy on data.
    if False:
        spy_images_from_raw(raw_train,get_label_name)
    train, validation, test = preprocess_data(raw_train,raw_validation,raw_test)
    #Lets see the final data how looks like.
    print(train)
    print(validation)
    print(test)

    #Now the shape is different.
    for img, label in raw_train.take(2):
        print("Original shape:", img.shape)

    for img, label in train.take(2):
         print("New shape:", img.shape)


    
    
if __name__ == '__main__':
    run()