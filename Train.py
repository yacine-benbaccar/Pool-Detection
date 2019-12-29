from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array, array_to_img
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from matplotlib import pyplot as plt
from PoolNet import PoolNetBaseline
import warnings
import os

warnings.filterwarnings("ignore")

class TrainModel:
    def __init__(self, model:Sequential, *args):
        self.CHECKPOINTPATH = './data/best_weights_baseline.h5'
        self.VALDIR = './data/validation'
        self.TRAINDIR = './data/train'
        self.IMGWIDTH, self.IMGHEIGHT = 50, 50
        self.CHANNELS = 3
        self.class1 = '/pools'
        self.class2 = '/no_pools'

        self.graphPath = './data/acc_loss_history.png'
        self.batchSize = 16 # default values will be changed later
        self.nbEpochs = 50 # default values will be changed later
        self.nbValidation = len([1 for f in os.listdir(self.VALDIR+self.class1) 
            if os.path.isfile(os.path.join(self.VALDIR+self.class1, f))]) + \
            len([1 for f in os.listdir(self.VALDIR+self.class2) 
                if os.path.isfile(os.path.join(self.VALDIR+self.class2, f))]) 
        self.nbTrain = len([1 for f in os.listdir(self.TRAINDIR+self.class1) 
                if os.path.isfile(os.path.join(self.TRAINDIR, f))]) + \
            len([1 for f in os.listdir(self.TRAINDIR+self.class2) 
                if os.path.isfile(os.path.join(self.TRAINDIR+self.class2, f))])
        
        self.trainGenerator = None
        self.valGenerator = None
        self.modelHistory = None

        self.model = model
    
    def dataAugmentation(self)->None:
        # Data augmentation: Transformation to perform on input
        # images to make the classifier more robust to noisier
        # and lower quality input images that might be given during
        # the testing phase

        # Normal data augmentation
        trainDataGen = ImageDataGenerator(
            rotation_range=90,  #default value 180
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            rescale=1./255,
            zoom_range=0.3, # default value 0.2
            horizontal_flip=True,
            fill_mode='nearest'
        )
        # Aggressive data augmentation
        # trainDataGen = ImageDataGenerator(
        #     rotation_range=180,
        #     width_shift_range=0.5,
        #     height_shift_range=0.5,
        #     shear_range=0.2,
        #     rescale=1./255,
        #     zoom_range=0.3,
        #     horizontal_flip=True,
        #     fill_mode='nearest'
        # )
        # For the validation process the only transformation to make is
        # the rescaling of the imag to avoid altering the input
        valDataGen = ImageDataGenerator(
            rescale=1./255
        )

        self.trainGenerator = trainDataGen.flow_from_directory(
            self.TRAINDIR,
            target_size=(self.IMGHEIGHT, self.IMGWIDTH),
            batch_size=self.batchSize,
            class_mode='binary'
        )
        self.valGenerator = valDataGen.flow_from_directory(
            self.VALDIR,
            target_size=(self.IMGHEIGHT, self.IMGWIDTH),
            batch_size=self.batchSize,
            class_mode='binary'
        )
        return


    def trainModel(self, useAutoAugmentation:bool=True, trackPerf:bool=True)->None:
        if useAutoAugmentation:
            self.dataAugmentation()
        # Keras Callbacks:
        # EarlyStopping: Used to stop the training when there is no more gains
        #   in terms of loss (decreasing) and accuracy (increasing)
        # ModelCheckpoint: Used to save the weights of the best performing model
        #   on the validation dataset
        callbacks = [
            ModelCheckpoint(self.CHECKPOINTPATH, monitor='val_loss', save_best_only=True),
            EarlyStopping(monitor='val_loss', patience=5)
        ]

        self.model.compile(
            loss='binary_crossentropy',
            optimizer='adam', # try it with rmsprop
            metrics=['accuracy']
        )
        self.modelHistory = self.model.fit_generator(
            self.trainGenerator,
            steps_per_epoch=self.nbTrain // self.batchSize,
            epochs=self.nbEpochs,
            validation_data=self.valGenerator,
            validation_steps=self.nbValidation // self.batchSize,
            callbacks=callbacks
        )
        if trackPerf:
            self.__trackModel__()

        return


    def __trackModel__(self)->None:
        # Plot the evolution of the loss and accuracy for training and validation
        # through the epochs
        epochs = range(len(self.modelHistory.history['acc']))
        _, ax1 = plt.subplots()
        ax1.plot(epochs, self.modelHistory.history['loss'], label='train_loss', color='blue')
        ax1.plot(epochs, self.modelHistory.history['val_loss'], label='val_loss', color='red')
        ax1.set_xlabel("# Epochs")
        ax1.set_ylabel('Loss')
        ax1.legend()

        ax2 = ax1.twinx()
        ax2.plot(epochs, self.modelHistory.history['acc'], label='train_acc', color='green')
        ax2.plot(epochs, self.modelHistory.history['val_acc'], label='val_acc', color='orange')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        plt.title('Loss and Accuracy during the Learning Process')
        plt.savefig(self.graphPath)
        return

if __name__ == "__main__":
    width, height, channels = 50, 50, 3
    model = PoolNetBaseline.build(width, height, channels)
    
    trainHelper = TrainModel(model)
    trainHelper.trainModel()
