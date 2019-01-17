import numpy as np
from PIL import Image
import glob
import tensorflow as tf
import keras
from keras import optimizers, applications
from keras.models import Model,Sequential
from keras import backend as K
import pandas as pd
from keras.layers.core import Lambda, Dense
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, Conv2DTranspose, Concatenate, Add, Multiply, ZeroPadding2D
from keras.layers.convolutional import AtrousConvolution2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from skimage.transform import resize
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, array_to_img, load_img
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm
import sys
from sklearn.model_selection import StratifiedKFold


# Function to upsize images
def upsize(img):
    return resize(img, (128, 128, 1), mode='constant', preserve_range=True)

# Function to downsize images
def downsize(img):
    return resize(img, (101, 101, 1), mode='constant', preserve_range=True)
