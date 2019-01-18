import argparse
import configparser
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import img_to_array, array_to_img, load_img
from keras import optimizers, applications
from keras.models import Model
from preprocessing import *
from utils import *
from model import *
from bestThres_utils import *

config = configparser.ConfigParser()
config['Parameters'] = {}


parser = argparse.ArgumentParser()
parser.add_argument("depths_path", help="path to depths.csv file here")
parser.add_argument("trainset_path", help="path to train set directory here")
args = parser.parse_args()


depths = pd.read_csv(args.depths_path)
for root, dirs, files in os.walk(args.trainset_path, topdown=False):
  
  X = np.zeros([len(files),128,128,1])
  Y = np.zeros([len(files),128,128,1])
  depths_train = np.zeros([len(files),128,128,1])
  
  for i in tqdm(range(X.shape[0])):
    
    filename = files[i].replace('.png','')
    
    depths_train[i,:,:,:] = depths.loc[depths['id']==filename, 'z'].values / np.max(depths['z'])
    
    img = load_img(root+'/'+files[i])
    X[i] = upsize(np.array(img))
    
    img = load_img(root.replace('images','masks')+'/'+files[i])
    Y[i] = upsize(np.array(img)/255)
     
    
# Data augmentation & shuffle
X = np.concatenate((X, [flip(img, orientation = 'leftright') for img in X], [flip(img, orientation = 'updown') for img in X]), axis=0)
Y = np.concatenate((Y, [flip(img, orientation = 'leftright') for img in Y], [flip(img, orientation = 'updown') for img in Y]), axis=0)
depths_train = np.concatenate((depths_train, depths_train, depths_train), axis=0)

X, Y, depths_train = shuffle(X, Y, depths_train, random_state=2018)


# Hyperparameters for the model 
callbacks = [
    EarlyStopping(patience=19, verbose=1),
    ReduceLROnPlateau(patience=5, verbose=1, factor=0.5),
    ModelCheckpoint('model-tgs-salt-1.h5', monitor='val_IOU_metric',verbose=1, save_best_only=True, save_weights_only=True, mode='max')
]

opt = keras.optimizers.Nadam(lr=0.01)

# Compile and Train
model = modified_GCN()
model.compile(optimizer=opt, loss=iou_bce_loss, metrics=[IOU_metric])
model.fit(x=[X,depths_train], y=Y, batch_size=32, epochs=200, callbacks=callbacks, validation_split=0.2)



# Best Threshold searching
thresholds = np.linspace(0.3,0.8,50)
ious = np.array([iou_metric_batch(val_y, np.int32((output_valid) > threshold)) for threshold in tqdm(thresholds, total = len(thresholds) )])
threshold_best_index = np.argmax(ious)
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

# Visualize the threshold vs ious graph
plt.plot(thresholds, ious)
plt.plot(threshold_best, iou_best, 'rx', label = 'Best threshold')
plt.xlabel('Thresholds')
plt.ylabel('IoU')
plt.title('Thresholds vs IoU ({},{})'.format(threshold_best, iou_best))
plt.legend()


# Save best threshold to the config file
config['Parameters']['threshold_best'] = str(threshold_best)
with open('config.ini', 'w') as configfile:
  config.write(configfile)
