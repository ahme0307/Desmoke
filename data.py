from __future__ import print_function

import os
import numpy as np
import pdb
import cv2
from fnmatch import fnmatch
from skimage.io import imsave, imread
import pickle
import pylab
import imageio
import matplotlib.pyplot as plt
import pylab
import imageio
import matplotlib.pyplot as plt
#Prepare training and test set
def create_train_data(param):
    filenames_img = []
    filenames_mask = []
    Gpaths_gt=param.orig_path
    if  os.path.exists('imgs_trainPath.npy')==True and os.path.exists('imgs_mask_trainPath.npy')==True :
        print('Training set already exists and loaded from file')
        return
    data_path=param.data_path
    Gpaths=[x for x in next(os.walk(data_path))][1]
    Gpaths=[os.path.join(data_path,x) for x in Gpaths]

    images = os.listdir(data_path)
    total =sum(len(os.listdir(os.path.join(y, os.path.basename(os.path.normpath(y))))) for y in (Gpaths))
    i = 0
    print('-'*30)
    print('Creating trainig images...')
    print('-'*30)
    img_mask=[]
    for video_number in range(len(images)):
        for imagename in os.listdir(os.path.join(Gpaths[video_number],images[video_number])):
            mask_nameRoot,ext =os.path.splitext(imagename)
            #pdb.set_trace()
            if os.path.exists(os.path.join(Gpaths_gt, '%s%s' %(mask_nameRoot[:-2],ext))):
                temp=os.path.join(Gpaths_gt, '%s%s' %(mask_nameRoot[:-2],ext))
            elif os.path.exists(os.path.join(Gpaths_gt, '%s%s' %(mask_nameRoot,ext))):
                temp=os.path.join(Gpaths_gt, '%s%s' %(mask_nameRoot,ext))
            else:
                print("File not found!!")
                pdb.set_trace()
                return

            try:
                filenames_img.append(os.path.join(Gpaths[video_number],images[video_number], imagename))
                filenames_mask.append(temp)
            except ValueError:
                pdb.set_trace()

            if i % 1000 == 0:
                print('Done: {0}/{1} images'.format(i, total))
            i += 1
            if i == total:
                print('Loading done.')
                np.save('imgs_trainPath.npy', filenames_img)
                np.save('imgs_mask_trainPath.npy', filenames_mask)
                print('Saving to .npy files done.')
                print('Loading done.')
                return

def load_train_data():
    imgs_train = np.load('imgs_trainPath.npy')
    imgs_mask_train = np.load('imgs_mask_trainPath.npy')
    return imgs_train, imgs_mask_train



def plot_imagesT(images, cls_true, cls_pred=None, smooth=True, filename='test.png'):
    #pdb.set_trace()
    assert len(images) == len(cls_true)
    fig, axes = plt.subplots(4, 4,figsize=(60, 60))
   
    if cls_pred is None:
        hspace = 0.6
    else:
        hspace = 0.9
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'
    count1 =0
    count2 =0
    for i, ax in enumerate(axes.flat):
        if i < len(images)*2:
            # Plot image.
            if i % 2 ==0:
                ax.imshow(np.uint8(images[count1]),interpolation=interpolation)
                count1+= 1
            else:
                ax.imshow(np.uint8(cls_true[count2]),interpolation=interpolation)
                count2+= 1
        ax.set_xticks([])
        ax.set_yticks([])
  #  plt.rcParams["figure.figsize"] = (60,60)
    plt.savefig(filename,dpi=100)
    plt.show()
