
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
keras = tf.keras
IMG_SIZE = 160

def spy_images_from_raw(train_image,train_label):
    # display 2 images from the dataset
    for image, label in train_image.take(5):
        plt.figure()
        plt.imshow(image)
        plt.title(train_label(label))
    plt.show()


def load_dataset_pretrained():
    '''
    The model we are going to use as the convolutional base for our model is the **MobileNet V2**
    developed at Google. This model is trained on 1.4 million images and has 1000 different classes.
    We want to use this model but only its convolutional base. 
    So, when we load in the model, we'll specify that we don't want to load the top (classification)
    layer. We'll tell the model what input shape to expect and to use the predetermined 
    weights from *imagenet* (Googles dataset).
    '''
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')
    base_model.summary()
    return base_model


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
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label


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


def plot_results(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


def run():
    
    TRAIN = False
    PREDICT = True

    #Download and get partitions. get_label_name is a method, recive a number and get the label.
    raw_train, raw_validation, raw_test, get_label_name = load_dataset()
    
    #Spy on data.
    if False:
        spy_images_from_raw(raw_train,get_label_name)
    train, validation, test = preprocess_data(raw_train,raw_validation,raw_test)
    class_names = [get_label_name(i) for i in range(0,2)]
    print(class_names)
    #Lets see the final data how looks like.
    print(train)
    print(validation)
    print(test)

    #Now the shape is different.
    for img, label in raw_train.take(2):
        print("Original shape:", img.shape)

    for img, label in train.take(2):
        print("New shape:", img.shape)

    if TRAIN:
        #load a pretrained model. and use it to classify cats vs dogs.
        base_model = load_dataset_pretrained()
        for image, _ in train.take(1):
            feature_batch = base_model(image)
            print(feature_batch.shape) #(32, 5, 5, 1280)

        #We dont want to train all the parametres, so we freeze it.
        base_model.trainable = False
        base_model.summary() #Trainable parameters = 0

        #Now we must add our classification layer.
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        #and the prediction layer. We add a dense layer because we only have 2 classes to predict.
        prediction_layer = keras.layers.Dense(1)
        #and finally we create the model.
        model = tf.keras.Sequential([
                base_model,
                global_average_layer,
                prediction_layer
                ])
        os.system("clear")
        model.summary()

        #Set learning rate hyperparameter.
        base_learning_rate = 0.0001
        #Define optimizer, loss function, and metrics evaluation/training.
        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
        
        # We can evaluate the model right now to see how it does 
        # before training it on our new images
        initial_epochs = 3
        validation_steps=20
        loss0,accuracy0 = model.evaluate(validation, steps = validation_steps)
        
        # Now train the model with out dataset.
        # Now we can train it on our images
        history = model.fit(train,
                            epochs=initial_epochs,
                            validation_data=validation)
        
        plot_results(history)
        
        # we can save the model and reload it at anytime in the future
        model.save("dogs_vs_cats.h5") 

    if PREDICT: 
        new_model = tf.keras.models.load_model('dogs_vs_cats.h5')
        predictions = new_model.predict(test)
     
        # Retrieve a batch of images from the test set
        image_batch, label_batch = test.as_numpy_iterator().next()
        predictions = new_model.predict_on_batch(image_batch).flatten()

        # Apply a sigmoid since our model returns logits
        predictions = tf.nn.sigmoid(predictions)
        predictions = tf.where(predictions < 0.5, 0, 1)

        print('Predictions:\n', predictions.numpy())
        print('Labels:\n', label_batch)
        print('Classes:\n',class_names)
        plt.figure(figsize=(10, 10))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            image = tf.cast((image_batch[i]*127.5 +1), tf.uint8) 
            plt.imshow(image)
            plt.title(class_names[predictions[i]])
            plt.axis("off")
        plt.show()
   
    
if __name__ == '__main__':
    run()
