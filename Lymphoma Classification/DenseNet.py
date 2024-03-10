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
img_dtype = tables.UInt8Atom()  # dtype in which the images will be saved, this indicates that images will be saved as unsigned int 8 bit, i.e., [0,255]
filenameAtom = tables.StringAtom(itemsize=255) #create an atom to store the filename of the image, just incase we need it later, 
cwd = os.getcwd()
files=glob.glob(cwd+"/lymphoma/*/*")
phases={}
phases["train"],phases["val"]=next(iter(model_selection.ShuffleSplit(n_splits=1,test_size=0.1).split(files)))
input('')
filters=tables.Filters(complevel=6, complib='zlib')
block_shape=[128,128,3]
stride=128;
classes=["FL","CLL","MCL"]

def get_submatrices(layer,size,s=1):
      
      a=layer
      Hout = (a.shape[1] - size[0]) // s + 1
      Wout = (a.shape[2] - size[1]) // s + 1
      Stride = (a.strides[0], a.strides[1] * s, a.strides[2] * s, a.strides[1], a.strides[2], a.strides[3])
 
      a = as_strided(a, (a.shape[0], Hout, Wout, size[0], size[1], a.shape[3]), Stride)
      return a
storage={
}

for phase in phases:
    hdf5_file = tables.open_file(f"./{datasetName}_{phase}.pytable", mode='w') #open the respective pytable
    storage["filenames"] = hdf5_file.create_earray(hdf5_file.root, 'filenames', filenameAtom, (0,)) #create the array for storage

    storage["imgs"]= hdf5_file.create_earray(hdf5_file.root, "imgs", img_dtype,  
                                              shape=np.append([0],block_shape), 
                                              chunkshape=np.append([1],block_shape),
                                              filters=filters)
    storage["labels"]= hdf5_file.create_earray(hdf5_file.root, "labels", img_dtype,  
                                              shape=[0], 
                                              chunkshape=[1],
                                              filters=filters)
    
    for i in range(len(phases[phase])):
        fname=files[phases[phase][i]];
        id=[idx for idx in range(len(classes)) if classes[idx] in fname][0]
        io=np.array(cv2.cvtColor(cv2.imread(fname),cv2.COLOR_BGR2RGB));
        io=io.reshape(1,io.shape[0],io.shape[1],io.shape[2])
        io_out=get_submatrices(io,(block_shape[0],block_shape[1]),stride)

        io_out=io_out.reshape(-1,block_shape[0],block_shape[1],block_shape[2]);
        storage["imgs"].append(io_out);
        #print(io_out.shape)
        storage["labels"].append([id for _ in range(io_out.shape[0])])
        print(len(storage["imgs"]))
    hdf5_file.close();
