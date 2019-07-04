import numpy as np
import pdb
from skimage.io import imsave, imread
import cv2
import os
from skimage.transform import resize
from skimage.io import imsave
#import matplotlib.pyplot as plt
import random
from skimage.transform import rotate
from skimage import transform
import math
def adjust_gamma(image , gamma):
    invgamma = 1 / gamma
    table = np.array([ ((i / 255.0 )**invgamma)*255 for i in np.arange(0,256)]).astype("float32")
    return cv2.LUT(image, table)

def add_gaussian_noise(X_imgs):
    row, col,_= X_imgs.shape
    X_imgs = X_imgs.astype(np.float32)
    mean = 0
    var = 0.1
    sigma = var ** 0.5    
    gaussian = np.random.random((row, col, 1)).astype(np.float32)
    gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
    gaussian_img = cv2.addWeighted(X_imgs, 0.75, 0.25 * gaussian, 0.25, 0)
    gaussian_img = np.array(gaussian_img, dtype = np.uint8)
    return gaussian_img
def load_images(image_paths,mask_paths):
    img=[]
    mask=[]
    shapes=[]
    for img_path, mask_path in zip(image_paths,mask_paths):
        tempimg,tempmask,shapesinfo=load_image_Mask(img_path,mask_path)
        img.append(tempimg)
        mask.append(tempmask)
        shapes.append(shapesinfo)

    return np.asarray(img),np.asarray(mask),shapes

def load_image_Mask(img_path,mask_path):
    img = imread(img_path)
    img_mask= imread(mask_path)
    rw=img.shape[0]
    cl=img.shape[1]
    ch=np.shape(img.shape)[0]
    img= cv2.resize(img, (img_rows,img_cols))
    img_mask =  cv2.resize(img_mask,(img_rows,img_cols))
    img =np.squeeze(img[:,:,:3])
    img_mask =np.squeeze(img_mask[:,:,:3])
    return img,img_mask, (rw,cl,ch)

def load_image(image_path):
    img = imread(image_path)
    #(rw,cl)=img.shape
    rw=img.shape[0]
    cl=img.shape[1]
    ch=np.shape(img.shape)[0]
    if (rw,cl) != (image_rows,image_cols):
        img=cv2.resize(img, (image_rows,image_cols))
   
    return img

# guided filter
def guided_filter(data, num_patches, width, height):
    r = 3
    channel=3
    eps = 1.0
    batch_q = np.zeros((num_patches, height, width, channel))
    temp = np.zeros((num_patches, height, width, channel))
    for i in range(num_patches):
        for j in range(channel):
            I = data[i, :, :,j]
            p = data[i, :, :,j]
            ones_array = np.ones([height, width])
            N = cv2.boxFilter(ones_array, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0)
            mean_I = cv2.boxFilter(I, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            mean_p = cv2.boxFilter(p, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            mean_Ip = cv2.boxFilter(I * p, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            cov_Ip = mean_Ip - mean_I * mean_p
            mean_II = cv2.boxFilter(I * I, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            var_I = mean_II - mean_I * mean_I
            a = cov_Ip / (var_I + eps) 
            b = mean_p - a * mean_I
            mean_a = cv2.boxFilter(a , -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            mean_b = cv2.boxFilter(b , -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            q = mean_a * I + mean_b 
            batch_q[i, :, :,j] = q
        batch_q[i, :, :,:] =  cv2.normalize(batch_q[i, :, :,:], batch_q[i, :, :,:], 0, 1, cv2.NORM_MINMAX)
    return batch_q
def laplacian(y_batch):    
    A = np.squeeze(y_batch[0])
    G = A.copy()
    gp = [G]
    #pdb.set_trace()
    for i in range(1,5):
        gp.append(np.squeeze(y_batch[i]))
    lp = [gp[4]]
    r=64
    for i in range(4,0,-1):  
        GE=cv2.resize(gp[i], (r,r))
        #print(i)
        L =gp[i-1]-GE
        lp.append(L)
        r=2*r
    return lp

def get_Log(y_batch):
    LOG_512 = np.zeros((y_batch[0].shape[0], int(img_rows), int(img_cols), 3))
    LOG_256 = np.zeros((y_batch[0].shape[0], int(img_rows/2), int(img_cols/2), 3))
    LOG_128 = np.zeros((y_batch[0].shape[0], int(img_rows/4), int(img_cols/4), 3))
    LOG_64 =  np.zeros((y_batch[0].shape[0], int(img_rows/8), int(img_cols/8), 3))
    LOG_32 =  np.zeros((y_batch[0].shape[0], int(img_rows/16), int(img_cols/16), 3))
    for k in range(y_batch[0].shape[0]):
        lpA=laplacian((y_batch[0][k],y_batch[1][k],y_batch[2][k],y_batch[3][k],y_batch[4][k]))
        #pdb.set_trace()
        LOG_32[k,:,:,:]=lpA[0]
        LOG_64[k,:,:,:]=lpA[1]
        LOG_128[k,:,:,:]=lpA[2]
        LOG_256[k,:,:,:]=lpA[3]
        LOG_512[k,:,:,:]=lpA[4]
    return LOG_512,LOG_256,LOG_128,LOG_64,LOG_32

def batch_process(x_batch, param, test_batch_size=None):
    global img_rows
    global img_cols
    img_rows=param.img_rows
    img_cols=param.img_cols
    test_batch_size=param.test_batch_size
    num_images =len(x_batch)

    x_batch = x_batch.astype('float32')
    x_batch/=255.
    x_batch=np.reshape(x_batch,(-1,img_rows,img_cols,3))
   # pdb.set_trace()
    #Pyramid
    
    temp = np.zeros((x_batch.shape[0], int(img_rows/2), int(img_cols/2), 3))
    for k in range(x_batch.shape[0]):
        #pdb.set_trace()
        temp[k] = cv2.resize(blur[k], (int(img_rows/2), int(img_cols/2)), interpolation = cv2.INTER_NEAREST)
    blur1=guided_filter(temp,num_patches = x_batch.shape[0], width = temp.shape[1], height = temp.shape[2])
    temp = np.zeros((x_batch.shape[0], int(img_rows/4), int(img_cols/4), 3))
    for k in range(x_batch.shape[0]):
        temp[k] = cv2.resize(blur1[k], (int(img_rows/4), int(img_cols/4)), interpolation = cv2.INTER_NEAREST)
    normalized2,blur=guided_filter(temp,num_patches = x_batch.shape[0], width = temp.shape[1], height = temp.shape[2])
    temp = np.zeros((x_batch.shape[0], int(img_rows/8), int(img_cols/8), 3))
    for k in range(x_batch.shape[0]):
        temp[k] = cv2.resize(blur2[k], (int(img_rows/8), int(img_cols/8)), interpolation = cv2.INTER_NEAREST)
    normalized3,blur=guided_filter(temp,num_patches = x_batch.shape[0], width = temp.shape[1], height = temp.shape[2])
    temp = np.zeros((x_batch.shape[0], int(img_rows/16), int(img_cols/16), 3))
    for k in range(x_batch.shape[0]):
        temp[k] = cv2.resize(blur3[k], (int(img_rows/16), int(img_cols/16)), interpolation = cv2.INTER_NEAREST)
    blur4=guided_filter(temp,num_patches = x_batch.shape[0], width = temp.shape[1], height = temp.shape[2])
   
    #pdb.set_trace()
    LOG_512,LOG_256,LOG_128,LOG_64,LOG_32=get_Log((x_batch,blur1,blur2,blur3,blur4))
    y_batch = y_batch.astype('float32')  
    y_batch /= 255.  # scale masks to [0, 1]
    y_batch=np.reshape(y_batch,(-1,img_rows,img_cols,3))

    return (x_batch,LOG_256,LOG_128,LOG_64,LOG_32),y_batch

def random_batch(image_train,imgs_mask_train,param, train_batch_size=None,x_valid_batch=None,y_valid_batch=None):
    num_images =len(image_train)
    global img_rows
    global img_cols
    img_rows=param.img_rows
    img_cols=param.img_cols
    if x_valid_batch is None and train_batch_size is None :
        x_batch = image_train
        y_batch = imgs_mask_train
    elif x_valid_batch is None:
 
        ids_train_batch = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)
        x_batch = image_train[ids_train_batch]
        y_batch = imgs_mask_train[ids_train_batch]
    else:
        x_batch = x_valid_batch
        y_batch = y_valid_batch
    
    x_batch,y_batch,shapesinfo=load_images(x_batch,y_batch)
    x_batch = x_batch.astype('float32')
    x_batch/=255.
    x_batch=np.reshape(x_batch,(-1,img_rows,img_cols,3))
  
    temp = np.zeros((x_batch.shape[0], 256, 256, 3))
    for k in range(x_batch.shape[0]):
        #pdb.set_trace()
        temp[k] = cv2.resize(x_batch[k], (256,256), interpolation = cv2.INTER_NEAREST)
        
    blur1=guided_filter(temp,num_patches = x_batch.shape[0], width = temp.shape[1], height = temp.shape[2])
    temp = np.zeros((x_batch.shape[0], 128, 128, 3))
    for k in range(x_batch.shape[0]):
        temp[k] = cv2.resize(blur1[k], (128,128), interpolation = cv2.INTER_NEAREST)
    blur2=guided_filter(temp,num_patches = x_batch.shape[0], width = temp.shape[1], height = temp.shape[2])
    temp = np.zeros((x_batch.shape[0], 64, 64, 3))
    for k in range(x_batch.shape[0]):
        temp[k] = cv2.resize(blur2[k], (64,64), interpolation = cv2.INTER_NEAREST)
    blur3=guided_filter(temp,num_patches = x_batch.shape[0], width = temp.shape[1], height = temp.shape[2])
    temp = np.zeros((x_batch.shape[0], 32, 32, 3))
    for k in range(x_batch.shape[0]):
        temp[k] = cv2.resize(blur3[k], (32,32), interpolation = cv2.INTER_NEAREST)
    blur4=guided_filter(temp,num_patches = x_batch.shape[0], width = temp.shape[1], height = temp.shape[2])
    #pdb.set_trace()
    LOG_512,LOG_256,LOG_128,LOG_64,LOG_32=get_Log((x_batch,blur1,blur2,blur3,blur4))
    y_batch = y_batch.astype('float32')  
    y_batch /= 255.  # scale masks to [0, 1]
    y_batch=np.reshape(y_batch,(-1,img_rows,img_cols,3))

    return (x_batch,LOG_256,LOG_128,LOG_64,LOG_32),y_batch,shapesinfo

def test_batch(image_train):
    #pdb.set_trace()
    img = [load_image(path) for path in image_train]
    img= np.squeeze(preprocess(np.asarray(img)))
    #pdb.set_trace()
    x_batch = preprocess(img)
    x_batch = x_batch.astype('float32')

    x_batch/=255.
    return np.squeeze(x_batch)

def plot_images(images, cls_true, cls_pred=None, smooth=True, class_names=None,filename='Encodertest.png'):
    #pdb.set_trace()
    assert len(images) == len(cls_true)
    fig, axes = plt.subplots(3, 3,figsize=(30, 30))

    # Adjust vertical spacing.
    if cls_pred is None:
        hspace = 0.8
    else:
        hspace = 1.2
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        # There may be less than 9 images, ensure it doesn't crash.
        if i < len(images):
            # Plot image.
            
            ax.imshow(np.uint8(images[i]),
                      interpolation=interpolation)

            # Name of the true class.
            cls_true_name = class_names[cls_true[i]]

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                # Name of the predicted class.
                cls_pred_name = class_names[cls_pred[i]]

                xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.savefig(filename,dpi=100)
    plt.show()
def get_layer_out(model,input_img_data, layer_name):
    input_img = model.input
    layer_output = model.get_layer(layer_name).output

    iterate = K.function([model.input], [layer_output])
    out_value = iterate([input_img_data])
    plt.imshow(np.squeeze(out_value))
    plt.show()
    return np.squeeze(out_value)