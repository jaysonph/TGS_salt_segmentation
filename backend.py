import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, Concatenate, Add


# Convolution layer, BatchNormalization and RELU activation
def Conv_Batch_Relu(x, output_channel_size, f1, f2, s=1, p='same', activation=False, rate =1):
  act = Conv2D(output_channel_size, (f1,f2), strides=(s,s), padding=p, dilation_rate=rate)(x)
  act = BatchNormalization()(act)
  if activation:
    act = Activation('relu')(act)
  return act

# Full Pre-activation Residual Block
def ResBlock(x, output_channel_size):
  a_shortcut = Conv2D(output_channel_size, (3,3), strides=(1,1), padding='same')(x)
  a_shortcut = BatchNormalization()(a_shortcut)
  a = BatchNormalization()(x)
  a = Activation('relu')(a)
  a = Conv2D(output_channel_size, (3,3), strides=(1,1), padding='same')(a)
  a = BatchNormalization()(a)
  a = Activation('relu')(a)
  a = Conv2D(output_channel_size, (3,3), strides=(1,1), padding='same')(a)
  a_out = Add()([a, a_shortcut])
  return a_out

# GCN Block
def GCN_Block(x, k, output_channel_size):
  a_k1 = Conv_Batch_Relu(x, output_channel_size, k, 1)
  a_k1 = Conv_Batch_Relu(a_k1, output_channel_size, k, 1)
  a_1k = Conv_Batch_Relu(x, output_channel_size, 1, k)
  a_1k = Conv_Batch_Relu(a_1k, output_channel_size, 1, k)
  a = Add()([a_k1, a_1k])
  return a 

# Boundary Refinement Block
def BR_Block(x):
  a = Conv_Batch_Relu(x, 21, 3, 3, activation=True)
  a = Conv_Batch_Relu(a, 21, 3, 3)
  a = Add()([x, a])
  return a

# Atrous Spatial Pyramid Pooling
def ASPP(x, c):
  x1 = Conv_Batch_Relu(x, c, 1, 1, activation=False)
  x2 = Conv_Batch_Relu(x, c, 3, 3, activation=False, rate =2)
  x3 = Conv_Batch_Relu(x, c, 3, 3, activation=False, rate =6)
  x4 = Conv_Batch_Relu(x, c, 3, 3, activation=False, rate =12)
  x5 = Conv_Batch_Relu(x, c, 3, 3, activation=False, rate =18)
  x6 = Conv_Batch_Relu(x, c, 3, 3, activation=False, rate =24)
  out = Concatenate()([x1,x2,x3,x4,x5,x6])
  return out
