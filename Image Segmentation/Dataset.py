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
from helper import get_submatrices
import json 
from constants import block_shape,stride
datasetName="epistroma"
img_dtype = tables.UInt8Atom()  # dtype in which the images will be saved, this indicates that images will be saved as unsigned int 8 bit, i.e., [0,255]
filenameAtom = tables.StringAtom(itemsize=255) #create an atom to store the filename of the image, just incase we need it later, 
#cwd+"/epi
cwd = os.getcwd()
DATSETPTH=cwd+"/epi/"
files=glob.glob(cwd+"/epi/masks/*.png")
print(files);
phases={}
phases["train"],phases["val"]=next(iter(model_selection.ShuffleSplit(n_splits=1,test_size=0.1).split(files)))


filters=tables.Filters(complevel=6, complib='zlib')
# classes=["FL","CLL","MCL"]


storage={
}

for phase in phases:
    hdf5_file = tables.open_file(f"./{datasetName}_{phase}.pytable", mode='w') #open the respective pytable

    storage["imgs"]= hdf5_file.create_earray(hdf5_file.root, "imgs", img_dtype,  
                                              shape=np.append([0],block_shape), 
                                              chunkshape=np.append([1],block_shape),
                                              filters=filters)
    storage["labels"]= hdf5_file.create_earray(hdf5_file.root, "labels", img_dtype,  
                                              shape=np.append([0],block_shape), 
                                              chunkshape=np.append([1],block_shape),
                                              filters=filters)
    if (phase=="val"):
        with open('indices.json', 'w') as f:
            json.dump(list(map(int,phases[phase])), f)
    for i in range(len(phases[phase])):
        fname=files[phases[phase][i]];
        #id=[idx for idx in range(len(classes)) if classes[idx] in fname][0]
        #print(fname);
        #print(cv2.imread("./data/"+os.path.basename(fname).replace("_mask.png",".tif")),cv2.COLOR_BGR2RGB)
        img=cv2.cvtColor(cv2.imread((DATSETPTH+os.path.basename(fname)).replace("_mask.png",".tif")),cv2.COLOR_BGR2RGB)
        img=img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
        img_out=get_submatrices(img,(block_shape[0],block_shape[1]),stride)
        img_out=img_out.reshape(-1,block_shape[0],block_shape[1],block_shape[2]);

        io=np.array(cv2.cvtColor(cv2.imread(fname),cv2.COLOR_BGR2RGB));
        io=io.reshape(1,io.shape[0],io.shape[1],io.shape[2])
        io_out=get_submatrices(io,(block_shape[0],block_shape[1]),stride)
        print(io_out.shape);
        io_out=io_out.reshape(-1,block_shape[0],block_shape[1],block_shape[2]);
        storage["imgs"].append(img_out)
        storage["labels"].append(io_out);
        #print(io_out.shape)
        #storage["labels"].append([id for _ in range(io_out.shape[0])])
        print(len(storage["imgs"]))
    hdf5_file.close();
