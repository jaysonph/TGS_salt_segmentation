import numpy as n
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, Concatenate, Add
from backend import *

def modified_GCN():
  
  pic = Input(shape=(128,128,1))
  depths = Input(shape=(128,128,1))
  
  ## pic & depths combining layer
  pic_conv = Conv2D(1, (3,3), strides=(1,1), padding='same')(pic)
  pic_conv = BatchNormalization()(pic_conv)
  depths_conv = Conv2D(1, (3,3), strides=(1,1), padding='same')(depths)
  depths_conv = BatchNormalization()(depths_conv)
  inputs = Add()([pic_conv, depths_conv])

  ## Res layer
  res2 = ResBlock(MaxPooling2D((2, 2))(inputs), 64)
  res3 = ResBlock(MaxPooling2D((2, 2))(res2), 256)
  res4 = ResBlock(MaxPooling2D((2, 2))(res3), 512)
  res5 = ResBlock(MaxPooling2D((2, 2))(res4), 1024)

  ## GCN layer
  GCN1 = GCN_Block(res2, 7, 21)
  GCN2 = GCN_Block(res3, 7, 21)
  GCN3 = GCN_Block(res4, 7, 21)
  GCN4 = GCN_Block(res5, 7, 21)

  ## BR layer 1
  BR1 = BR_Block(GCN1)
  BR2 = BR_Block(GCN2)
  BR3 = BR_Block(GCN3)
  BR4 = BR_Block(GCN4)

  ## Deconv1 layer
  deconv1 = Conv2DTranspose(21, (3,3), strides=(2,2), activation='relu', padding='same')(BR4)

  ## Deconv2 layer
  pre_deconv2 = Add()([BR3, deconv1])
  pre_deconv2 = BR_Block(pre_deconv2)
  deconv2 = Conv2DTranspose(21, (3,3), strides=(2,2), activation='relu', padding='same')(pre_deconv2)

  ## Deconv3 layer
  pre_deconv3 = Add()([BR2, deconv2])
  pre_deconv3 = BR_Block(pre_deconv3)
  deconv3 = Conv2DTranspose(21, (3,3), strides=(2,2), activation='relu', padding='same')(pre_deconv3)

  ## Deconv4 layer
  pre_deconv4 = Add()([BR1, deconv3])
  pre_deconv4 = BR_Block(pre_deconv4)
  deconv4 = Conv2DTranspose(21, (3,3), strides=(2,2), activation='relu', padding='same')(pre_deconv4)


  ## Score Map
  outputs = BR_Block(deconv4)
  outputs = ASPP(outputs, 16)
  outputs = BatchNormalization()(outputs)
  outputs = Activation('relu')(outputs)
  outputs = Conv_Batch_Relu(outputs, 32, 1, 1, activation=True, rate =1)
  outputs = Conv2D(1, (1,1), strides=(1,1), padding='valid')(outputs)
  outputs = BatchNormalization()(outputs)
  outputs = Activation('sigmoid')(outputs)
  
  model = Model(inputs=[pic,depths], outputs=outputs)
  
  return model
