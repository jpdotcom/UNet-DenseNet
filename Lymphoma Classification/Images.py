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

datasetName="Lymphoma"
classes=["FL","CLL","MCL"]


with tables.open_file(f"./{datasetName}_{"train"}.pytable", mode='r') as db:
    for (i,data) in enumerate(db.root.imgs):
        plt.title(classes[db.root.labels[i]]);
        plt.imshow(data)
        plt.show()