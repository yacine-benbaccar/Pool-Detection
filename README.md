# Pool-Detection

The objectif for this project is to provide a POC (Proof Of Concept) to prove the feasability of **detecting pools from satellite images**.

This task will be divided into two steps :

- Create a image classification model (Convnet) to distinguish between images with *pools* and images without *no_pools*.
- Then use this model to scan satellite images and find the position of potential pools on the image.

## Available Data

## Installing the required python packages

```console
pip install -r requirements.txt
```

## Models

### Baseline Model

The baseline model is a simple 3-layered Convnet. This model is a simple implementation of a *Convolutional Neural Network* and will be used as refrence (in terms of performance) to the rest of the tested models.

## Results

### Quality of the classifier

![alt text](https://github.com/yacine-benbaccar/Pool-Detection/blob/master/data/acc_loss_history.png "Train/Validation Loss/Accuracy")

### Quality of the detection

![alt text](https://github.com/yacine-benbaccar/Pool-Detection/blob/master/data/detected/pooldetection_th%3D0.75_zone1.jpg "Detection on the first satellite image")