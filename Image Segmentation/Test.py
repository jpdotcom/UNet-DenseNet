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
import tables
import json 
from helper import get_submatrices
from constants import blockshape
device=torch.device('mps')
classes=["FL","CLL","MCL"]
cwd = os.getcwd()


img_size=1000;
stride=blockshape[0]
model=torch.load(cwd+"/model.pt")
w,r=0,0
files=glob.glob(cwd+"/epi/masks/*.png")
DATSETPTH=cwd+"/epi/"
def repatch(arr):
   #print(np.array(arr).shape);
   ## (9,256,256,3)
   #print(arr.shape);
   out=[]
   arr=np.array(arr);
   for i in range(arr.shape[3]):
      
      plane=arr[:,:,:,i];
      plane=plane.reshape(img_size//plane.shape[1],img_size//plane.shape[2],plane.shape[1],plane.shape[2])
      plane=np.concatenate(plane,axis=1);
      print(plane.shape);
      plane=np.concatenate(plane,axis=1)
      print(plane.shape);
      out.append(plane);
   
   out=np.array(out).transpose([1,2,0])
   return out
def getImages(display=False):

  test_images,test_labels,test_out=[],[],[]
  with tables.open_file(cwd+"/epistroma_val.pytable",'r') as db:

    with open("indices.json",'r') as f:
          
          indicies=json.load(f)
          
          #arr=np.random.randint(0,len(db.root.imgs), 1000)

          for (i,idx) in enumerate(indicies):
            fname=files[idx];

            fullimg=cv2.cvtColor(cv2.imread((DATSETPTH+os.path.basename(fname)).replace("_mask.png",".tif")),cv2.COLOR_BGR2RGB)
            fullimg=fullimg.reshape(1,fullimg.shape[0],fullimg.shape[1],fullimg.shape[2])
            fullimg=get_submatrices(fullimg,(blockshape[0],blockshape[1]),stride);
            fullimg=fullimg.reshape(-1,blockshape[0],blockshape[1],blockshape[2]);
            
            
            fulllabel=cv2.cvtColor(cv2.imread(fname),cv2.COLOR_BGR2RGB)
            #plt.imshow(fulllabel);
            #lt.show();
            fulllabel=fulllabel.reshape(1,fulllabel.shape[0],fulllabel.shape[1],fulllabel.shape[2])
            fulllabel=get_submatrices(fulllabel,(blockshape[0],blockshape[1]),stride);
            fulllabel=fulllabel.reshape(-1,blockshape[0],blockshape[1],blockshape[2]);

            total_out=[]
            total_img=[];
            total_labels=[];
            for i in range(fullimg.shape[0]):
              label=fulllabel[i]
              img=fullimg[i];

              
              #plt.imshow(img);
              #plt.show();
              img=np.transpose(img,[2,0,1]);
              img=torch.tensor(img);
                  #print(torch.tensor(img,dtype=torch.float)/255)
              img=img.to(device)
              out=model((torch.tensor(img,dtype=torch.float)/255).reshape(1,img.shape[0],img.shape[1],img.shape[2])).cpu().detach();
              out=torch.nn.functional.sigmoid(out);
              out=out.numpy();
              img = img.cpu().detach().numpy();
              #print(np.max(out ));
              #outE=np.exp(out[0]);
              #pred=np.argmax(outE/np.sum(outE))
              #check= pred ==  db.root.labels[i];
        
              out=out[0]
              img=np.transpose(img,[1,2,0])
              out=np.transpose(out,[1,2,0])
              total_out.append(out);
              total_img.append(img);
              total_labels.append(label);
              
            #total_img=np.concatenate(total_img,axis=(0));
            #total_out=np.concatenate(total_out,axis=(0));
            #print(np.array(total_img).shape);
            total_img=repatch(total_img);
            
            total_out=repatch(total_out);
            total_out=np.rint(total_out);
            total_labels=repatch(total_labels
                                )
            f,axis=plt.subplots(3);
            if (display):
               
              axis[0].imshow(total_img);
              axis[1].imshow(total_out);
              axis[2].imshow(total_labels)
              plt.show();
            test_images.append(total_img);
            test_labels.append(total_labels)
            test_out.append(total_out);
          #img=img.cpu();
          #img=np.transpose(img,[1,2,0])
          #title=f"Expect value: {classes[db.root.labels[i]]}. \n Predicted value: {classes[pred]}"
          # plt.title(title);
          # plt.imshow(img);
          # plt.show();
          # if (check):
          #     r+=1
          # else:
          #     w+=1 
          
          # print(f"Right queries: {r}. Wrong queries: {w}")
  return test_images,test_labels,test_out

if (__name__ == "__main__"):
   getImages(True);
