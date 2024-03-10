import numpy as np
import tables
import os,sys
import glob
import PIL
from numpy.lib.stride_tricks import as_strided
import cv2
import matplotlib.pyplot as plt
from sklearn import model_selection
import random

datasetName="epistroma"
classes=["FL","CLL","MCL"]


with tables.open_file(f"./{datasetName}_{"train"}.pytable", mode='r') as db:
    for (i,data) in enumerate(db.root.imgs):
        
        #plt.title(classes[db.root.labels[i]]);
        f,axarr = plt.subplots(2,2)

        axarr[0,0].imshow(data)
        axarr[0,1].imshow(db.root.labels[i])
        plt.show()