import tables
import os,sys
import glob
import PIL
import numpy as np
from numpy.lib.stride_tricks import as_strided
import cv2
import matplotlib.pyplot as plt
from sklearn import model_selection 
import random
import torch
from torch.utils.data import DataLoader
from torchvision.models import DenseNet
from torch import nn
from albumentations import *
from albumentations.pytorch import ToTensorV2


device=torch.device('mps')
classes=["FL","CLL","MCL"]
cwd=os.getcwd();
def getImages(display=False):
        
    model=torch.load(cwd+"/model.pt")
    w,r=0,0
    tot_images,tot_out,tot_label=[],[],[]
    with tables.open_file(cwd+"/Lymphoma_val.pytable") as db:
        arr=np.random.randint(0,len(db.root.imgs), 1000)
        
        for i in arr:
            
            
            img=db.root.imgs[i];

            img=np.transpose(img,[2,0,1]);
            img=torch.tensor(img);
                #print(torch.tensor(img,dtype=torch.float)/255)
            img=img.to(device)
            out=model(2*((torch.tensor(img,dtype=torch.float)/255)-0.5).reshape(1,img.shape[0],img.shape[1],img.shape[2])).cpu().detach().numpy();
            outE=np.exp(out[0]);
            pred=np.argmax(outE/np.sum(outE))
            check= pred ==  db.root.labels[i];
            
            img=img.cpu();
            img=np.transpose(img,[1,2,0])
            title=f"Expect value: {classes[db.root.labels[i]]}. \n Predicted value: {classes[pred]}"
            if (display):
                plt.title(title);
                plt.imshow(img);
                plt.show();
            if (check):
                r+=1
            else:
                w+=1 
            
            print(f"Right queries: {r}. Wrong queries: {w}")
            tot_images.append(img);
            tot_out.append(pred);
            tot_label.append(db.root.labels[i]);
    
    return tot_images,tot_label,tot_out;

if (__name__=="__main__"):
    getImages(True);