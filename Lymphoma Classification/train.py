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
import torch
from torch.utils.data import DataLoader
from torchvision.models import DenseNet
from torch import nn
from albumentations import *
from albumentations.pytorch import ToTensorV2
num_classes=3    #number of classes in the data mask that we'll aim to predict
in_channels= 3  #input channel of the data, RGB = 3
num_epochs=100;
batch_size=64
growth_rate=64 
block_config=(2, 2, 2, 2)
num_init_features=100
bn_size=4
patch_size=100;
drop_rate=0
device =torch.device("mps");
cwd=os.getcwd();
#patch_size=224;
model = DenseNet(growth_rate=growth_rate, block_config=block_config,
                 num_init_features=num_init_features, 
                 bn_size=bn_size, 
                 drop_rate=drop_rate, 
                 num_classes=num_classes).to(device)
#model=torch.load("/Users/jaypatel/Desktop/INVent Lab /model.pt")
model.to(device);
print(model);
#print(model);
img_transform = Compose([
       VerticalFlip(p=.5),
       HorizontalFlip(p=.5),
       HueSaturationValue(hue_shift_limit=(-25,0),sat_shift_limit=0,val_shift_limit=0,p=1),
       Rotate(p=1, border_mode=cv2.BORDER_CONSTANT,value=0),
       RandomSizedCrop((patch_size,patch_size), patch_size,patch_size),
       ToTensorV2()
    ])
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
            #img=np.transpose(img,[2,0,1]);
            img_new = img_transform(image=self.imgs[index])['image']
            #print(img_new.shape)
            #plt.imshow(np.transpose(img_new,[1,2,0]));
            #plt.show();
            #print(torch.tensor(img,dtype=torch.float)/255)
            return 2*((torch.tensor(img_new,dtype=torch.float)/255)-0.5), label;
            #return torch.tensor(self.img_transform(image=img)['image'],dtype=torch.float), label
    
    def __len__(self):
        return self.len

# img_transform = Compose([
#        VerticalFlip(p=.5),
#        HorizontalFlip(p=.5),
#        HueSaturationValue(hue_shift_limit=(-25,0),sat_shift_limit=0,val_shift_limit=0,p=1),
#        Rotate(p=1, border_mode=cv2.BORDER_CONSTANT,value=0),
#        #ElasticTransform(always_apply=True, approximate=True, alpha=150, sigma=8,alpha_affine=50),
#        RandomSizedCrop((patch_size,patch_size), patch_size,patch_size),
#        ToTensorV2()
#     ])
weights=[0,0,0];
with tables.open_file(cwd+"/Lymphoma_train.pytable") as db:
      for p in db.root.labels:
          weights[p]+=1

weights=((1-torch.tensor(weights))/sum(weights)).to(device);
        
datasets={"train": Dataset(cwd+"/Lymphoma_train.pytable"),
          "val": Dataset(cwd+"/Lymphoma_val.pytable")}
dataLoader={"train": DataLoader(datasets["train"], batch_size=batch_size, 
                                shuffle=True, num_workers=0,pin_memory=True),
            
            "val": DataLoader(datasets["val"], batch_size=batch_size,shuffle=True, num_workers=0,pin_memory=True)
            } 

# (img, label)=datasets["train"][7]
# fig, ax = plt.subplots(1,1, figsize=(10,4))  # 1 row, 2 columns

# #build output showing patch after augmentation and original patch
# ax.imshow();
# plt.show();


optim = torch.optim.Adam(model.parameters())
criteron=nn.CrossEntropyLoss(weight=weights)
min_loss=float('inf');
for i in range(num_epochs):
    L=0
    model.train();
    with torch.set_grad_enabled(True):
        for j, (img, label) in enumerate(dataLoader["train"]):
            #print(img.shape);
            #img.dtype=torch.float;
            #img/=255;
            img=img.to(device);
            #plt.imshow(np.transpose(img[0].cpu(),(1,2,0)));
            #plt.show();
            label = label.type('torch.LongTensor').to(device)
        
            pred=model(img);
        
            loss=criteron(pred,label);
            optim.zero_grad();
            loss.backward();
            optim.step();
            L += loss.item() 
            
    print(f"Train loss: {L}")
    L=0;

    model.eval();
    with torch.no_grad():

        for j, (img, label) in enumerate(dataLoader["val"]):
        #print(img.shape);
        #img.dtype=torch.float;
        #img/=255;
            img=img.to(device);
            label = label.type('torch.LongTensor').to(device)
            
            pred=model(img);

            loss=criteron(pred,label);
            L += loss.item()
    
    if (L<min_loss):
        min_loss=L;
        torch.save(model, cwd+"/model.pt")
        print("Model saved");
    
    print(f"Val Loss {L}");

#torch.save(model, "/Users/jaypatel/Desktop/INVentLab/Image Segmentation/model.pt")

