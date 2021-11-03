import tensorflow as tf
import numpy as np

def predict_with_model(model, img_path):
#
    #Read file.
    image = tf.io.read_file(img_path)
    #Decode file to png.
    image = tf.image.decode_png(image, channels=3)
    #Convert pixels to float32 type. Rescale values to 0 - 1. 
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    #Resize image (train, validation and test size)
    image = tf.image.resize(image, [60,60])
    #Expand dimension (add batch size)
    image = tf.expand_dims(image, axis=0)

    #Obtain prediction probabilities 
    predictions = model.predict(image)
    #Obtain the higest probability index.
    predictions = np.argmax(predictions)

    return predictions

if __name__ == "__main__":

    img_path = "/home/lautaro/Cursos/Introduction To Tensorflow2/GermanTrafficSignRecorgnitionBenchmark/Test/0/00807.png"
    model = tf.keras.models.load_model('./Models')
    prediction = predict_with_model(model, img_path)
    print(f"prediction = {prediction}")