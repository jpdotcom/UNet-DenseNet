import torch
from torch import nn
from torch.utils.data import DataLoader

from albumentations import *
from albumentations.pytorch import ToTensorV2


from unet import UNet #code borrowed from https://github.com/jvanvugt/pytorch-unet

import PIL
import matplotlib.pyplot as plt
import cv2

import numpy as np
import sys, glob


import scipy.ndimage 

import time
import math
import tables
import os
import random
from constants import block_shape

cwd = os.getcwd()

# +
device=torch.device("mps")
dataname="epistroma"
ignore_index = -100 #Unet has the possibility of masking out pixels in the output image, we can specify the index value here (though not used)
gpuid=0

# --- unet params
#these parameters get fed directly into the UNET class, and more description of them can be discovered there
n_classes= 3   #number of classes in the data mask that we'll aim to predict
in_channels= 3  #input channel of the data, RGB = 3
padding= True   #should levels be padded
depth= 5       #depth of the network 
wf= 2           #wf (int): number of filters in the first layer is 2**wf, was 6
up_mode= 'upconv' #should we simply upsample the mask, or should we try and learn an interpolation 
batch_norm = True #should we use batch normalization between the layers

# --- training params
batch_size=128
patch_size=block_shape[0]
num_epochs = 100
edge_weight = 1.1 #edges tend to be the most poorly segmented given how little area they occupy in the training set, this paramter boosts their values along the lines of the original UNET paper
phases = ["train","val"] #how many phases did we create databases for?
img_transform = Compose([
       VerticalFlip(p=.5),
       HorizontalFlip(p=.5),
       #HueSaturationValue(hue_shift_limit=(-25,0),sat_shift_limit=0,val_shift_limit=0,p=1),
       Rotate(p=1, border_mode=cv2.BORDER_CONSTANT,value=0),
       #ElasticTransform(always_apply=True, approximate=True, alpha=150, sigma=8,alpha_affine=50),
       RandomSizedCrop((patch_size,patch_size), patch_size,patch_size),
       ToTensorV2()],
       additional_targets={'label': 'image'}
    )
class Dataset(object):
    def __init__(self,fname) -> None:
        self.fname=fname
        with tables.open_file(self.fname,'r') as db:

            self.len=len(db.root.imgs);
        self.imgs=None
        self.labels=None;
    def __getitem__(self,index):
        with tables.open_file(self.fname,'r') as db:

            self.imgs=db.root.imgs

            self.labels=db.root.labels;
            img=self.imgs[index]
            label=self.labels[index]
            #print(img.shape);
            #print(img.shape);
            transform=img_transform(image=img, label=label)
            img=transform['image']
            label=transform['label'];
            
            #img=np.transpose(img,[2,0,1]);
            #label=np.transpose(label,[2,0,1])
            #img_new = img_transform(image=self.imgs[index])['image']
            #print(img_new.shape)
            #plt.imshow(np.transpose(img,[1,2,0]));
            #plt.show();
            #print(torch.tensor(img,dtype=torch.float)/255)
            img_new=img;
            return torch.tensor(img_new,dtype=torch.float)/255, torch.tensor(label,dtype=torch.float)/255;
            #return torch.tensor(self.img_transform(image=img)['image'],dtype=torch.float), label
    
    def __len__(self):
        return self.len

datasets={"train": Dataset(cwd+"/epistroma_train.pytable"),
          "val": Dataset(cwd+"/epistroma_val.pytable")
          }
dataLoader={"train": DataLoader(datasets["train"], batch_size=batch_size, 
                                shuffle=True, num_workers=0,pin_memory=True),
            
            "val": DataLoader(datasets["val"], batch_size=batch_size,shuffle=True, num_workers=0,pin_memory=True)
            } 

model = UNet(n_classes=n_classes, in_channels=in_channels, padding=padding,depth=depth,wf=wf, up_mode=up_mode, batch_norm=batch_norm).to(device)
model = torch.load(cwd+"/model.pt")
criteron = nn.MSELoss(reduce=False) #reduce = False makes sure we get a 2D output instead of a 1D "summary" value
optim = torch.optim.Adam(model.parameters())
min_val=float('inf');
for i in range(num_epochs):



    model.train();
    L=0;
    
    for (j,(img,out)) in enumerate(dataLoader["train"]):

        img=img.to(device);
        out=out.to(device);
        #f,axarr=plt.subplots(2);

   
        
        pred=model(img);
        pred=torch.nn.functional.sigmoid(pred)

        with torch.set_grad_enabled(True):


            loss_matrix=criteron(pred,out);
            
            loss=loss_matrix.mean();
            optim.zero_grad();
            loss.backward();
            optim.step();
            L+=loss.item();
        print(f"Batch {j} Done ")

    print(f"Loss: {L}")
    L=0;
    model.eval();
    with torch.no_grad():

        for j, (img, img_out) in enumerate(dataLoader["val"]):
        #print(img.shape);
        #img.dtype=torch.float;
        #img/=255;
            img=img.to(device);
            img_out = img_out.to(device)
            
            pred=model(img);
            pred=torch.nn.functional.sigmoid(pred);

            loss_matrix=criteron(pred,img_out);
            loss=loss_matrix.mean();
            L += loss.item() 
    print(f"Val Loss {L}");
    if (L<min_val):
        torch.save(model,cwd+"/model.pt")
        min_val=L;
        print("Model saved");
    
  
    
    L=0;

#torch.save(model,cwd+"model.pt")