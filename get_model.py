
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose,BatchNormalization,Activation,UpSampling2D,Flatten, Dense,AveragePooling2D,add,AveragePooling2D,add,Dropout,ZeroPadding2D,Convolution2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.optimizers import RMSprop
from keras.losses import binary_crossentropy

from keras.models import Sequential
import cv2
from keras.legacy import interfaces
from keras.optimizers import Optimizer
from keras.utils import get_file
from sklearn.metrics import mean_squared_error


def PSNRLoss(y_true, y_pred):
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)

def psnr(y_true, y_pred):
    return -10. * np.log10(np.mean(np.square(y_pred - y_true)))
def conv_block(input_tensor, filters, stage, block, strides=(2, 2)):
    conv_name_base = 'ColNet_' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_uniform')(input_tensor)
    x = BatchNormalization(name=bn_name_base)(x)
    x = Activation('relu')(x) 
    
    return x

def desmokeNET(param, num_classes=3):
    img_rows=param.img_rows
    img_cols=param.img_cols
    img_ch=3
    inputs_Orig = Input((img_rows, img_cols, img_ch))
    inputs_L1   = Input((img_rows/2, img_cols/2, img_ch))
    inputs_L2   = Input((img_rows/4, img_cols/4, img_ch))
    inputs_L3   = Input((img_rows/8, img_cols/8, img_ch))
    inputs_L4   = Input((img_rows/16, img_cols/16, img_ch))
    
   
    inputs=[inputs_Orig,inputs_L1,inputs_L2,inputs_L3,inputs_L4]
    # 512
    down0a=conv_block(inputs_Orig, 32, 512, 'input', strides=(2, 2))
    down0a=conv_block(down0a, 32, 512, 'root', strides=(2, 2))
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    
    # 256_side
    down0_sidex=conv_block(inputs_L1, 32, 256, 'input', strides=(2, 2))   
    # 256_join
    conc_down0=concatenate([down0a_pool,down0_sidex], axis=3)
    down0=conv_block(conc_down0, 64, 256, 'root', strides=(2, 2))
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
   
    # 128_side
    down_sidex=conv_block(inputs_L2, 32, 32, '128_input', strides=(2, 2))   
    #128_join
    conc_down1=concatenate([down0_pool,down_sidex], axis=3)
    down1=conv_block(conc_down1, 128, 128, 'root', strides=(2, 2))
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    
    #64_side
    down_sidex=conv_block(inputs_L3, 32, 64, '64_input', strides=(2, 2))   
    #64_join
    conc_down2=concatenate([down1_pool,down_sidex], axis=3)
    down2=conv_block(conc_down2, 256, 64, 'root', strides=(2, 2))
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32_side
    down_sidex=conv_block(inputs_L4, 32, 2, '32_input', strides=(2, 2))     
    #32_join
    conc_down3=concatenate([down2_pool,down_sidex], axis=3)
    down3=conv_block(conc_down3, 512, 32, 'root', strides=(2, 2))
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)

    # 16  # center

    center=conv_block(down3_pool, 1024, 16, 'center', strides=(2, 2))
    center=conv_block(center, 1024, 16, 'center_2', strides=(2, 2))
    # 32
    up4 = UpSampling2D((2, 2))(center)
    up4=conv_block(up4, 512, 32, 'deconv1', strides=(2, 2))
    up4 = concatenate([down3, up4],axis=3)
    up4=conv_block(up4, 512, 32, 'deconv2', strides=(2, 2))
    #up4 = Dropout(0.2)(up4)
    up4=conv_block(up4, 512, 32, 'deconv3', strides=(2, 2))
    
    # 64

    up3 = UpSampling2D((2, 2))(up4)
    up3=conv_block(up3, 256, 64, 'deconv1', strides=(2, 2))
    up3 = concatenate([down2, up3],axis=3)
    up3=conv_block(up3, 256, 64, 'deconv2', strides=(2, 2))
    #up3 = Dropout(0.2)(up3)
    up3=conv_block(up3, 256, 64, 'deconv3', strides=(2, 2))
    # 128

    up2 = UpSampling2D((2, 2))(up3)
    up2=conv_block(up2, 128, 128, 'deconv1', strides=(2, 2))
    up2 = concatenate([down1, up2],axis=3)
    up2=conv_block(up2, 128, 128, 'deconv2', strides=(2, 2))
   # up2 = Dropout(0.2)(up2)
    up2=conv_block(up2, 128, 128, 'deconv3', strides=(2, 2))
    # 256

    up1 = UpSampling2D((2, 2))(up2)
    up1=conv_block(up1, 64, 256, 'deconv1', strides=(2, 2))
    up1 = concatenate([down0, up1],axis=3)
    #up1 = Dropout(0.2)(up1)
    up1=conv_block(up1, 64, 256, 'deconv2', strides=(2, 2))
    up1 = Dropout(0.2)(up1)
    up1=conv_block(up1, 64, 256, 'deconv3', strides=(2, 2))


    # 512

    up0a = UpSampling2D((2, 2))(up1)
    up0a=conv_block(up0a, 32, 512, 'deconv1', strides=(2, 2))
    up0a = concatenate([down0a, up0a],axis=3)
    up0a=conv_block(up0a, 32, 512, 'deconv2', strides=(2, 2))
    up0a=conv_block(up0a, 32, 512, 'deconv3', strides=(2, 2))
  
    
    

    classify = Conv2D(3, (1, 1), activation='sigmoid')(up0a)

    model = Model(inputs=inputs, outputs=classify)
    adam = Adam(lr=1e-3)
    
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=[PSNRLoss])

    return model




