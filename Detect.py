from keras.models import load_model, Sequential
from keras.preprocessing.image import img_to_array, array_to_img, load_img
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import seaborn as sns
import numpy as np
import json
import time
import warnings
import os

sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})
warnings.filterwarnings("ignore")

class Detector:
    def __init__(self, model:Sequential, zonesPath:str):
        self.model = model
        self.zonesPath = zonesPath
        self.zones = [os.path.join(zonesPath, f) for f in os.listdir(zonesPath) 
            if os.path.isfile(os.path.join(zonesPath, f))]
        self.threshold = 0.5
        
        self.results = {}
    
    def removePrefix(self, s:str)->str:
        if s.startswith(self.zonesPath):
            return s[len(self.zonesPath):]
        return s

    def splitImage(self, img:list)->list:
        # Splits an image of size `satHeight x satWidth` into a list of subimages of size = (imgHeight,imgWidth)
        return [img[:, imgWidth*row:imgWidth*(row+1),imgHeight*col:imgHeight*(col+1),:]
            for row in range(satHeight//imgHeight) for col in range(satWidth//imgWidth)]

    def prepImages(self)->np.ndarray:
        # Splits all the images into `imgHeight x imgWidth` subimage of size = (imgHeight,imgWidth)
        # so we get one np.array for all satellite image to test (for easier and faster predictions
        # of the probabilities)
        imgs = []
        for z in self.zones:
            img = (img_to_array(load_img(z)) * (1./255))
            img = img.reshape((1,) + img.shape)
            imgs.extend(self.splitImage(img))
        return np.array(imgs).reshape((len(imgs),imgHeight,imgWidth,channels))

    def predictProba(self)->tuple:
        x = self.prepImages()
        probas = self.model.predict_proba(x, batch_size=batchSize).reshape(len(x))
        probas = probas.reshape((len(self.zones), (satHeight//imgHeight)*(satWidth//imgWidth)))
        return np.where(probas>=self.threshold, probas, 0.), dict(zip(self.zones, probas))

    def getAdjacentBoxes(self, coords:list)->list:
        # naive approach, this can be further optimized for high resolution
        # images with a high number of detected instances 
        adj = []
        for i in range(len(coords)):
            for j in range(i+1,len(coords)):
                q = self.adjacent(coords[i], coords[j])
                if  q:
                    adj.append([i,j,q])
                else:
                    continue
        return adj

    def adjacent(self, box1:tuple, box2:tuple)->bool:
        if box1[1]==box2[1] and (box1[0]+50 == box2[0] or box1[0]-50 == box2[0]) or \
            box1[0]==box2[0] and (box1[1]+50 == box2[1] or box1[1]-50 == box2[1]):
            return True
        return False

    def cleanProba(self, probas:np.ndarray)->dict:
        res = {}
        # add a test to detect two adjacent boxes
        # if y-adjacent center the boxes on the y-axis
        # if x-adjacent center the boxes on the x-axis
        for idx, p in enumerate(probas):
            p = p.reshape(satHeight//imgHeight, satWidth//imgWidth)
            x,y = np.nonzero(p)
            res[self.zones[idx]] = {
                "pos":list(zip(y*50,x*50)),
                "probas":[ float(p[row][col]) for row,col in list(zip(x,y))],
                "adjacentBoxes":self.getAdjacentBoxes(list(zip(y*50,x*50)))
                # check for adjacent boxes for better detection
            }
            # find adjacent boxes
            toRemove = {
                "pos":[],
                "probas":[]
            }
            predAgain = []
            for bb in res[self.zones[idx]]["adjacentBoxes"]:
                # get old values
                b1 = res[self.zones[idx]]["pos"][bb[0]]
                b2 = res[self.zones[idx]]["pos"][bb[1]]
                # stage for for modifications and new probabilities
                toRemove["pos"].extend([bb[0],bb[1]])
                toRemove["probas"].extend([bb[0],bb[1]])
                # newly better placed box
                newBox = ((b1[0]+b2[0])//2, (b1[1]+b2[1])//2)
                predAgain.append(newBox)
            res[self.zones[idx]]["probas"] = [float(res[self.zones[idx]]["probas"][i])
                for i in range(len(res[self.zones[idx]]["probas"])) if i not in set(toRemove["probas"])]
            res[self.zones[idx]]["pos"] = [(int(res[self.zones[idx]]["pos"][i][0]),int(res[self.zones[idx]]["pos"][i][1]))
                for i in range(len(res[self.zones[idx]]["pos"])) if i not in set(toRemove["pos"])]
            # compute the probability for the new box
            # needs to be optimized!!!
            image = (img_to_array(load_img(self.zones[idx]))*1./255)
            image = image.reshape((1,)+image.shape)
            for h,w in predAgain:
                x = image[:,w:w+50,h:h+50,:]
                pr = self.model.predict_proba(x)[0][0]
                if pr >= self.threshold:
                    res[self.zones[idx]]["pos"].append((int(h),int(w)))
                    res[self.zones[idx]]["probas"].append(float(pr))   
            res[self.zones[idx]]["nbPools"] = len(res[self.zones[idx]]["pos"])
            res[self.zones[idx]].pop("adjacentBoxes",None)
        self.results = res
        return res

    def drawBoxes(self)->None:
        for img in self.results:
            _, ax = plt.subplots(figsize=(20,10))
            ax.imshow(load_img(img))
            imgData = self.results[img]
            # draw the bounding boxes and the probability for each
            # classification
            for coord, proba in zip(imgData["pos"], imgData["probas"]):
                color = "cyan"
                if proba >= 0.85: color = "lime"
                if proba <0.75: color="crimson"
                # drawing the bounding boxes
                rect = patches.Rectangle(coord,imgWidth, imgHeight,
                    linewidth=2, edgecolor=color, facecolor=color, alpha=0.4)
                ax.add_patch(rect)
                # drawing the probabilities for each classification
                if coord[1]+60 >= satHeight:
                    xy = (coord[0],coord[1]-10)
                else:
                    xy = (coord[0],coord[1]+60)
                
                ax.annotate(
                    s="prob:{:.3f}".format(proba),
                    xy=xy, color=color,weight="bold", ha="center", va="center")
            plt.xticks([])
            plt.yticks([])
            plt.savefig("./predictions/images/pooldetection_th={}_{}".format(self.threshold, self.removePrefix(img)))
        return

    def drawHeatmap(self, probaMap:list)->None:
        # cmap = sns.cubehelix_palette(light=1, as_cmap=True, reverse=False)
        cmap = sns.color_palette("Paired")
        # not the best implementation for it
        def f(i:int,j:int)->float:
            # fading function
            fadingRate = 1./(20*np.sqrt(2)) # 24*np.sqrt(2) default value
            # fadingRate = 0 # Debugging Value
            return max([1 - fadingRate * (np.sqrt((24-i)**2+(24-j)**2)),0])
        # the fadingAgent is used to create the fading effect on the heatmap
        fadingAgent = np.array([[f(i,j) for i in range(50)] for j in range(50)])
        for img in self.zones:
            # Loading the image on which we will overlay the heatmap
            _, ax = plt.subplots(figsize=(20,10))
            image = load_img(img)
            ax.imshow(image)
            # Potential targets to mark the potential positions of pools
            potTargets = [[],[]] # [xs:list, ys:list]
            tmpMap = probaMap[img].reshape(satHeight//imgHeight, satWidth//imgWidth)
            # scale up the probability matrix
            newProbaMap = np.zeros((800,1600))
            for i in range(satHeight//imgHeight):
                for j in range(satWidth//imgWidth):
                    newProbaMap[i*50:(i+1)*50, j*50:(j+1)*50] = tmpMap[i][j] * fadingAgent
                    if tmpMap[i][j]>0.5:
                        potTargets[0].append(j*50 + 24)
                        potTargets[1].append(i*50 + 24)
            # overlay the heatmap
            heatmap = sns.heatmap(newProbaMap, ax=ax, cmap=cmap, linewidths=0.0)
            heatmap.collections[0].set_alpha(0.1)
            # mark the potential positions for the pools in the area
            plt.scatter(potTargets[0], potTargets[1], marker='x', s=40, c='red')
            # discard the x and y labels
            plt.xticks([])
            plt.yticks([])
            # save the new heatmap
            plt.savefig("./predictions/images/heatmaps/heatmap_{}".format(self.removePrefix(img)))

        return

if __name__ == "__main__":
    # global vars
    satWidth, satHeight = 1600, 800
    imgWidth, imgHeight = 50, 50
    channels = 3
    batchSize = 16

    # load the model: PoolNet from the h5 file
    path2Weights = "./predictions/weights/best_weights_baseline_3.h5"
    trainFile = "Train.py"

    # check if the weights exist
    # load the model if the file exist otherwise train the model
    # using the trainHelper method implemented in `Train.py`
    if not os.path.isfile(path2Weights):
        print("---Training The Model---")
        os.system("python {}".format(trainFile))
        print("---Model Trained---")
        
    print("---Loading the Model---")
    model = load_model(path2Weights)
    print("---Model Loaded---")

    # Load the satellite images
    zonesPath = "./data/zones/"

    detect = Detector(model, zonesPath)

    tic = time.clock()
    probas, probaMap = detect.predictProba()
    results = detect.cleanProba(probas)
    elapsedTime = time.clock() - tic
    
    # detect.drawBoxes()
    detect.drawHeatmap(probaMap)
    print("Elapsed Time [s]: {:.3f}".format(elapsedTime))
    # with open("predictions/results.json","w") as f:
    #     json.dump(results, f, indent=4)

    print("END")