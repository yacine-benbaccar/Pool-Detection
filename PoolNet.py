from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Input, AveragePooling2D
from keras.layers import Activation, Dropout, Dense, Flatten
from keras import applications
from keras.utils import plot_model
from keras import backend as K

class PoolNetBaseline:
    # this is a baseline, a simple convnet model used as
    # a proof of concept
    @staticmethod
    def build(width:int, height:int, channels:int)->Sequential:
        if K.image_data_format() == 'channels_first':
            inputShape = (channels, width, height)
        else:
            inputShape = (width, height, channels)
        
        model = Sequential()
        model.add(Conv2D(32, (3,3), input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (5,5))) # extra
        model.add(Activation('relu')) # extra
        model.add(MaxPooling2D(padding='same'))

        model.add(Conv2D(32, (3,3)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (5,5))) # extra
        model.add(Activation('relu')) # extra
        model.add(MaxPooling2D(padding='same'))

        model.add(Conv2D(32, (3,3)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (5,5))) # extra
        model.add(Activation('relu')) # extra
        model.add(MaxPooling2D(padding='same'))

        # at this stage the model outputs a 3D feature map of the image

        model.add(Flatten()) # converts the 3D feature map into 1D feature vector
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.1)) # added to limit overfitting
        model.add(Dense(1))
        model.add(Activation('sigmoid')) # return a probability for the True class
        # in quality of classification
        # (Pool), forces the output to be in [0,1]
        # print(model.summary())
        # now return the neural network
        return model

# Leveraging pre-trained model from keras.applications
# Performs poorly on low res images
class PoolNetTL:
    @staticmethod
    def build(width:int, height:int, channels:int)->Sequential:
        if K.image_data_format() == 'channels_first':
            inputShape = (channels, width, height)
        else:
            inputShape = (width, height, channels)
        inp = Input(shape=inputShape)
        model = applications.vgg16.VGG16(include_top=False, pooling='avg', input_tensor=inp)
        x = model.layers[-1].output

        x = Dense(64)(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x) # added to limit overfitting
        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        model = Model(inputs=inp, outputs=x)

        for i in range(len(model.layers)-2):
            model.layers[i].trainable = False
        # print(model.summary())
        return model

if __name__ == "__main__":
    model = PoolNetBaseline.build(50,50,3)
    plot_model(model, 'PoolNetBaseline.png', show_shapes=True, rankdir='LR')
    print('END')