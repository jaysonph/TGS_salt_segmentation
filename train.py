import argparse
import os
import numpy as np
from preprocessing import *

parser = argparse.ArgumentParser()
parser.add_argument("depths_path", help="path to depths.csv file here")
parser.add_argument("trainset_path", help="path to train set directory here")
args = parser.parse_args()


depths = pd.read_csv(args.depths_path)
for root, dirs, files in os.walk(args.trainset_path, topdown=False):
  
  X = np.zeros([len(files),128,128,1])
  Y = np.zeros([len(files),128,128,1])
  depths_train = np.zeros([len(files),128,128,1])
  
  for i in range(X.shape[0]):
    
    filename = files[i].replace('.png','')
    
    depths_train[i,:,:,:] = depths.loc[depths['id']==filename, 'z'].values / np.max(depths['z'])
    
    img = load_img(root+'/'+files[i])
    X[i] = upsize(np.array(img))
    
    img = load_img(root.replace('images','masks')+'/'+files[i])
    Y[i] = upsize(np.array(img)/255)
     
    
# Data augmentation & shuffle
X = np.concatenate((X, [np.fliplr(img) for img in X], [np.flipud(img) for img in X]), axis=0)
Y = np.concatenate((Y, [np.fliplr(img) for img in Y], [np.flipud(img) for img in Y]), axis=0)
depths_train = np.concatenate((depths_train, depths_train, depths_train), axis=0)

X, Y, depths_train = shuffle(X, Y, depths_train, random_state=1)
