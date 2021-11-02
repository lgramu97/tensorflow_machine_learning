import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import shutil
import csv
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def display_some_examples(examples, labels):
    #Create a figure with data examples.
    plt.figure(figsize=(10,10))

    #Grid 5 x 5 of 25 random img
    for i in range(25):
        #Choose random index images.
        idx = np.random.randint(0,examples.shape[0]-1)
        img = examples[idx]
        label = labels[idx]

        plt.subplot(5,5, i+1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img, cmap='gray')

    plt.show()

def split_data(path_to_data, path_to_save_train, path_to_save_validation, split_size=0.1):

    folders = os.listdir(path_to_data)

    for folder in folders:

        full_path = os.path.join(path_to_data, folder)
        image_paths = glob.glob(os.path.join(full_path, '*.png'))

        x_train, x_val = train_test_split(image_paths, test_size=split_size)

        for x in x_train:
            #x represents full path, we want the name of the image.

            path_to_folder = os.path.join(path_to_save_train, folder)
            if not os.path.isdir(path_to_folder):
                os.mkdir(path_to_folder)

            shutil.copy(x, path_to_folder)

        for x in x_val:
            #x represents full path, we want the name of the image.

            path_to_folder = os.path.join(path_to_save_validation, folder)
            if not os.path.isdir(path_to_folder):
                os.mkdir(path_to_folder)

            shutil.copy(x, path_to_folder)


def order_test_set(path_to_images, path_to_csv):
    #Construct a dictionary name_image: value_label. Pone cada una de las imagenes en una carpeta
    #asociada al label que corresponde en Test.csv. Las imagenes estan todas juntas, entonces las 
    #separo por carpeta según la etiqueta que indica ese archivo.

    #Open csv file.
    try:
        with open(path_to_csv, 'r') as csvfile:

            reader = csv.reader(csvfile, delimiter=',')

            for i, row in enumerate(reader):
                #We dont need the row 0 (WIDTH;WEIGHT;LABEL;NAME...)
                if i==0:
                    continue
                #Remove /Test and take only the name.
                img_name = row[-1].replace('Test/','')
                label = row[-2]

                path_to_folder = os.path.join(path_to_images,label)

                if not os.path.isdir(path_to_folder):
                    os.mkdir(path_to_folder)

                img_full_path = os.path.join(path_to_images,img_name)
                shutil.move(img_full_path,path_to_folder)

    except:
        print("[INFO] : Sorry, something go wrong. Error reading csv file")


def create_generators(batch_size, train_data_path, val_data_path, test_data_path):
    #Create generators for each part of our dataset.

    #First preprocess data. Maybe could apply transformations here.
   
    train_preprocessor = ImageDataGenerator(
        rescale= 1 / 255.,
        rotation_range=10,
        width_shift_range=0.1
    )
    #Warning: If use transformations, dont use the same preprocesor for test and validation. 
    test_preprocessor = ImageDataGenerator(
        rescale= 1 / 255.
    )
 
    train_generator = train_preprocessor.flow_from_directory(
        train_data_path,
        class_mode='categorical',
        target_size=(60,60),
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size
    )

    val_generator = test_preprocessor.flow_from_directory(
        val_data_path,
        class_mode='categorical',
        target_size=(60,60),
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size
    )

    test_generator = test_preprocessor.flow_from_directory(
        test_data_path,
        class_mode='categorical',
        target_size=(60,60),
        color_mode='rgb',
        shuffle=False,
        batch_size=batch_size
    )

    return train_generator, val_generator, test_generator