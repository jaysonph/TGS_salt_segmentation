import argparse
import configparser
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.preprocessing.image import img_to_array, array_to_img, load_img
from preprocessing import *
from model import *


parser = argparse.ArgumentParser()
parser.add_argument("depths_path", help="path to depths.csv file here")
parser.add_argument("testimg_path", help="path to test img here")
parser.add_argument("weight_path", help="path to model weights here")
args = parser.parse_args()


# Load best threshold
config = configparser.ConfigParser()
config.read('config.ini')
best_threshold = float(config['Parameters']['threshold_best'])


# Load img and depth
depths = pd.read_csv(args.depths_path)
depths_test = np.zeros([128,128,1])
 
filename = args.testimg_path.split('/')[-1].split('.')[0]

depths_test[:,:,:] = depths.loc[depths['id']==filename, 'z'].values / np.max(depths['z'])

img = load_img(args.testimg_path)
test_X[i] = upsize(np.array(img))


# Load model and weights
modified_GCN() = model
model.load_weights(arg.weights)


# Predict each img using the average of img, fliplr version and flipud version
output = model.predict([test_X,depths_test],  verbose = 1)
output_fliplr = model.predict([np.array([flip(x, orientation = 'leftright') for x in test_X]),depths_test], batch_size=32 ,verbose = 1)
output_fliplr = np.array([flip(a, orientation = 'leftright') for a in output_flip])
output_flipud = model.predict([np.array([flip(x, orientation = 'updown') for x in test_X]),depths_test], batch_size=32 ,verbose = 1)
output_flipud = np.array([flip(a, orientation = 'updown') for a in output_flip])
output = output + output_fliplr + output_flipud
output = output/3

img = downsize(output)
img = np.squeeze(img, axis=-1)
img = (img >= best_threshold)*255
plt.imshow(img)
