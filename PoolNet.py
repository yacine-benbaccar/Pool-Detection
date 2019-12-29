from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Dense, Flatten
from keras import applications
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
        model.add(MaxPooling2D(padding='same'))

        model.add(Conv2D(32, (3,3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(padding='same'))

        model.add(Conv2D(32, (3,3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(padding='same'))

        # at this stage the model outputs a 3D feature map of the image

        model.add(Flatten()) # converts the 3D feature map into 1D feature vector
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5)) # added to limit overfitting
        model.add(Dense(1))
        model.add(Activation('sigmoid')) # return a probability for the True class
        # in quality of classification
        # (Pool), forces the output to be in [0,1]

        # now return the neural network
        return model

class PoolNetResnet:
    @staticmethod
    def build(width:int, height:int, channels:int)->Sequential:
        if K.image_data_format() == 'channels_first':
            inputShape = (channels, width, height)
        else:
            inputShape = (width, height, channels)
        
        model = applications.VGG16(include_top=False, weights='imagenet')
        return None