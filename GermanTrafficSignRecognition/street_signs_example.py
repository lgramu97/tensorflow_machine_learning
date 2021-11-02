import os
import glob
from posixpath import basename
from sklearn.model_selection import train_test_split
import shutil
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

from my_utils import split_data, order_test_set, create_generators
from deeplearning_models import streetsings_model


if __name__ == '__main__':
    
    if False:
        #Prepare train-validation directory.
        path_training_data = '/home/lautaro/Cursos/Introduction To Tensorflow2/GermanTrafficSignRecorgnitionBenchmark/training_data'
        path_to_save_train = os.path.join(path_training_data,'train')
        path_to_save_val = os.path.join(path_training_data,'val')

        if not os.path.isdir(path_training_data):
            os.mkdir(path_training_data)
            os.mkdir(path_to_save_train)
            os.mkdir(path_to_save_val)

        path_to_data= '/home/lautaro/Cursos/Introduction To Tensorflow2/GermanTrafficSignRecorgnitionBenchmark/Train'

        split_data(path_to_data, path_to_save_train, path_to_save_val)
    
        #Prepare test directory.
        path_to_images = '/home/lautaro/Cursos/Introduction To Tensorflow2/GermanTrafficSignRecorgnitionBenchmark/Test'
        path_to_csv = '/home/lautaro/Cursos/Introduction To Tensorflow2/GermanTrafficSignRecorgnitionBenchmark/Test.csv'
        order_test_set(path_to_images,path_to_csv)
    
    path_training_data = '/home/lautaro/Cursos/Introduction To Tensorflow2/GermanTrafficSignRecorgnitionBenchmark/training_data'
    path_to_train = os.path.join(path_training_data,'train')
    path_to_val = os.path.join(path_training_data,'val')
    path_to_test = '/home/lautaro/Cursos/Introduction To Tensorflow2/GermanTrafficSignRecorgnitionBenchmark/Test'
    batch_size = 64
    epochs = 15
    lr = 0.0001

    train_generator, val_generator, test_generator = create_generators(batch_size, path_to_train, path_to_val, path_to_test)
    nbr_classes = train_generator.num_classes
    
    TRAIN=False
    TEST=True

    if TRAIN:
        #Create Saver to save checkpoints
        path_to_save_model = './Models'
        ckpt_saver = ModelCheckpoint(
                        path_to_save_model,
                        monitor='val_accuracy',
                        verbose=1,
                        save_best_only=True,
                        mode='max',
                        save_freq='epoch'
                    )

        #If after x epochs, the metric dont improve, stop.
        early_stop = EarlyStopping(monitor='val_accuracy',patience=10)
        
        #Create the model
        model = streetsings_model(nbr_classes)
        
        #Create optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=True)

        #loss function must be the same as loss generator.
        model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])

        #we can pass generator because we used flow_from_directory
        model.fit(train_generator,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=val_generator,
                callbacks=[ckpt_saver, early_stop]
                )

    if TEST:
        #Want to evaluate the model i saved.
        model = tf.keras.models.load_model('./Models')
        model.summary()

        #Evaluate model in validtation and test set.
        print("Evaluating validation set:")
        model.evaluate(val_generator)

        print("Evaluating test set:")
        model.evaluate(test_generator)