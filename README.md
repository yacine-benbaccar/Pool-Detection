# Pool-Detection

The objectif for this project is to provide a POC (Proof Of Concept) to prove the feasability of **detecting pools from satellite images**.

This task will be divided into two steps :

- Create a image classification model (Convnet) to distinguish between images with *pools* and images without *no_pools*.
- Then use this model to scan satellite images and find the position of potential pools on the image.

***

## Available Data

```tree
└───data
    ├───train
    │   ├───no_pools
    │   └───pools
    ├───validation
    │   ├───no_pools
    │   └───pools
    └───zones
```

To create the image classifier we dispose of around 1500 images of each class (*pools*, *no_pools*). This data is divided into **train** and **validation** sets.
The **train** and **test** set contains respectively  around 1400 and 100 images of each class.
Each image has a dimension of 50 x 50.

Sample images:

*pools* : ![alt text](./data/train/pools/img0.jpg) ![alt text](./data/train/pools/img23.jpg) ![alt text](./data/train/pools/img451.jpg) ![alt text](./data/train/pools/img151.jpg) ![alt text](./data/train/pools/img843.jpg) ![alt text](./data/train/pools/img1023.jpg) ![alt text](./data/train/pools/img109.jpg) ![alt text](./data/train/pools/img702.jpg) ![alt text](./data/train/pools/img999.jpg) ![alt text](./data/train/pools/img1123.jpg) ![alt text](./data/train/pools/img93.jpg) ![alt text](./data/train/pools/img782.jpg)

*no_pools* : ![alt text](./data/train/no_pools/img0.jpg) ![alt text](./data/train/no_pools/img23.jpg) ![alt text](./data/train/no_pools/img451.jpg) ![alt text](./data/train/no_pools/img151.jpg) ![alt text](./data/train/no_pools/img823.jpg) ![alt text](./data/train/no_pools/img102.jpg) ![alt text](./data/train/no_pools/img92.jpg) ![alt text](./data/train/no_pools/img702.jpg) ![alt text](./data/train/no_pools/img1333.jpg) ![alt text](./data/train/no_pools/img102.jpg) ![alt text](./data/train/no_pools/img192.jpg) ![alt text](./data/train/no_pools/img91.jpg)

As for the satellite images on which the detection of *pools* will be performed we have 30 of them in the **data/zones/** directory.

Sample satellite image *(zone18.jpg)*:

![alt text](data/zones/zone18.jpg)*zone18.jpg*

***

## Installing the required python packages

```console
pip install -r requirements.txt
```

***

## Models

### Baseline Model

The baseline model is a simple 3-layered Convnet. This model is a simple implementation of a *Convolutional Neural Network* and will be used as refrence (in terms of performance) to the rest of the tested models.

![alt text](data/README/PoolNetBaseline_3.png)*PoolNetBaseline Architecture*

### Detection Mechanisms

![alt text](data/README/decomp.png)*Satelite image decomposition*

![alt text](data/README/merging_adj.png)*Merging adjacent bounding boxes for better location prediction*
***

## Results

### Quality of the classifier

- Number of training data :
    - *pools* : 1398 (50x50) images
    - *no_pools*: 1325 (50x50) images
- Number of validation data:
    - *pools*: 179 (50x50) images
    - *no_pools*: 176 (50x50) images

![alt text](data/README/acc_loss_history_3.png)*Train/Validation Loss/Accuracy*

### Quality of the detection

Sample Detection image *(zone18.jpg)*:

![alt text](data/README/pooldetection_th%3D0.5_zone18.jpg)*Detection on a satellite image with a threshold of 0.5 (the default output of the **detect** class)*

![alt text](data/README/pooldetection_th%3D0.75_zone18.jpg)*Detection on a satellite image with a threshold of 0.75 on the probability of each patch*

Along with the image we provide a dictonary that contains all information relative to the position and probabilities of each bounding boxes (in this context we only keep the patches with a probability > 0.5, as the purpose of this project is to prove the feasability of such detection).

Data relative to the dection image above *(zone18.jpg)*:

**"The reason we have only 6 bounding boxes in the detected image above is that we have a applied a filter that will only display patches with probability >= 0.75. This is done only for purpose of testing and visualization, by default the **detect** class will display all potential pool patches (ie: probability >= 0.5)"**

```json
"./data/zones/zone18.jpg": {
        "pos": [
            [250,100],
            [1400,200],
            [1100,250],
            [900,500],
            [850,550],
            [1450,650],
            [975,50]
        ],
        "probas": [
            0.9920390248298645,
            0.7856220006942749,
            0.5162228345870972,
            0.9985809922218323,
            0.878549337387085,
            0.9921371936798096,
            0.9999746084213257
        ],
        "nbPools": 7
}
```

Each tuple in the *pos* list represents the (x,y) coordinates of the top left corner of the bounding box that represents a potential pool (to get the center of the bounding box we just add 25 to x and y as we use a (50x50) sliding window for the detection). For better detection, we try to improve the predicted position of the bounding boxes by merging two adjacent boxes as sometimes a *pool* could be in two slinding window. This approach helps to make better detections and get closer to the real number of pools in that snapshot.

### Heatmaps