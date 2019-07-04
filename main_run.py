from __future__ import print_function
import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import pdb
from skimage.io import imsave, imread
import cv2
#import pylab
import imageio
#import matplotlib.pyplot as plt
from  get_model import *
from natsort import natsorted
from os.path import splitext
#from keras.utils import plot_model
#import pylab as plt
from skimage.io import imsave, imread
from  gen_data import batch_process,guided_filter
from skimage.io import imsave
#set the source directory here
filepath='./Input/'
#set the destination directory
pred_dir='./Desmoked'
#output dimention
out_size=(360,288)
def load_images(image_paths):
    image_rows = 512
    image_cols = 512
    imgs_p = np.ndarray((np.shape(image_paths)[0], image_rows, image_cols,3), dtype=np.uint8)
    

    for img_path,ii in zip(image_paths,range(0, np.shape(image_paths)[0])):
        img = imread(img_path)
        img= cv2.resize(img, (512,512))
        imgs_p[ii,:,:,:]=img
    x_batch=np.reshape(imgs_p,(-1,image_rows,image_cols,3))
    return x_batch


if __name__ == "__main__":
    eval_batch=5
    imtest=[]
    model = get_unetArch2()
    filename='desmokeNET.hdf5'

    model.load_weights(filename, by_name=True)
    files = [f for f in os.listdir(filepath) if f[-3:] == 'png']
    files=natsorted(files, key=lambda y: y.lower())
    for i in range(0, len(files)):
        imtest.append(os.path.join(filepath,files[i]))
    for start in range(0, len(imtest), eval_batch):
        end = min(start + eval_batch, len(imtest))
        x_valid_batch = imtest[start:end]        
        x_batch=load_images(x_valid_batch)
        x_batch=batch_process(x_batch)
       
        pred_y_batch = model.predict([x_batch[0],x_batch[1],x_batch[2],x_batch[3],x_batch[4]], batch_size=eval_batch,verbose=1)
        for pred, name in zip(pred_y_batch,x_valid_batch):
            pred=np.squeeze(pred)
            #pdb.set_trace()
            path=os.path.join(pred_dir,splitext( os.path.basename(os.path.normpath(name)))[0]+'_pred'+'.png')
            pred= cv2.resize(pred,out_size )
            imsave(path, pred)