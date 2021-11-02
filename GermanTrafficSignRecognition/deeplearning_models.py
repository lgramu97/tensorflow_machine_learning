import tensorflow
from tensorflow.keras.layers import Conv2D, Input ,Dense, MaxPool2D, BatchNormalization, Flatten, GlobalAvgPool2D
from tensorflow.keras import Model

# functional approach: se crea una funcion que retorna un modelo deep learning.
def functional_model():
    my_input = Input(shape=(28,28,1)) #gray scale images 28x28
    x = Conv2D(filters=32,kernel_size=(3,3),activation='relu')(my_input) #32 filters of 3x3 and then Relu.
    x = Conv2D(64,(3,3),activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(128,(3,3),activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(256,(3,3),activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    
    x = GlobalAvgPool2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10,activation='softmax')(x)#Output layer, Classification layer.

    model = tensorflow.keras.Model(inputs=my_input,outputs=x)

    return model

# tensorflow.keras.Model: herencia de una clase y se reimplementa metodos
class MyCustomModel2(tensorflow.keras.Model):
    
    def __init__(self):
        super().__init__()

        self.conv1 = Conv2D(filters=32,kernel_size=(3,3),activation='relu') #32 filters of 3x3 and then Relu.
        self.conv2 = Conv2D(64,(3,3),activation='relu')
        self.maxpool1 = MaxPool2D()
        self.batchnorm1 = BatchNormalization()
        
        self.conv3 = Conv2D(128,(3,3),activation='relu')
        self.maxpool2 = MaxPool2D()
        self.batchnorm2 = BatchNormalization()
        
        self.conv4 = Conv2D(256,(3,3),activation='relu')
        self.maxpool3 = MaxPool2D()
        self.batchnorm3 = BatchNormalization()
        
        self.globalavg1 = GlobalAvgPool2D()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(10,activation='softmax')#Output layer, Classification layer.

    def call(self, my_input):

        x = self.conv1(my_input)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x)
        x = self.conv4(x)
        x = self.maxpool3(x)
        x = self.batchnorm3(x)
        x = self.globalavg1(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

class MyCustomModel(tensorflow.keras.Model):

    def __init__(self):
        super().__init__()

        self.conv1 = Conv2D(32, (3,3), activation='relu')
        self.conv2 = Conv2D(64, (3,3), activation='relu')
        self.maxpool1 = MaxPool2D()
        self.batchnorm1 = BatchNormalization()

        self.conv3 = Conv2D(128, (3,3), activation='relu')
        self.maxpool2 = MaxPool2D()
        self.batchnorm2 = BatchNormalization()

        self.globalavgpool1 = GlobalAvgPool2D()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(10, activation='softmax')  
     

    def call(self, my_input):

        x = self.conv1(my_input)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x)
        x = self.globalavgpool1(x)
        x = self.dense1(x)
        x = self.dense2(x) 

        return x    


def streetsings_model(nbr_classes):

    my_input = Input(shape=(60,60,3))

    x = Conv2D(filters=32,kernel_size=(3,3),activation='relu')(my_input) #32 filters of 3x3 and then Relu.
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(64,(3,3),activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(128,(3,3),activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    
    #x = Flatten()(x) #Experiments... 3200 params vs 128.
    x = GlobalAvgPool2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(nbr_classes,activation='softmax')(x)#Output layer, Classification layer.

    return Model(inputs=my_input,outputs=x)

if __name__ == '__main__':
    #Only for test propouse. Check architecture.
    model = streetsings_model(10)
    model.summary()