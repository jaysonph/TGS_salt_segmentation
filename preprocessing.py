import numpy as np
from skimage.transform import resize


# Function to upsize images
def upsize(img):
    return resize(img, (128, 128, 1), mode='constant', preserve_range=True)


# Function to downsize images
def downsize(img):
    return resize(img, (101, 101, 1), mode='constant', preserve_range=True)
    

# Augmentation by flipping images
def flip(img, orientation):
  if orientation = 'updown':
    return np.flipud(img)
  if orientation = 'leftright':
    return np.fliplr(img)
