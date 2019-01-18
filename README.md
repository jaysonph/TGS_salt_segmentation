# TGS_salt_segmentation
Image segmentation using Deep Learning. An algorithm that automatically and accurately identifies if a subsurface target is salt or not.


## Data distribution
The depths of training and test data are distributed as below.

<img width="400" alt="2019-01-18 8 12 13" src="https://user-images.githubusercontent.com/40629085/51386565-d7ec3200-1b5d-11e9-8006-a665a400cf16.png">

## Preprocessing on data
1. Resize the data from 101x101 to 128x128

<img width="740" alt="2019-01-18 8 11 59" src="https://user-images.githubusercontent.com/40629085/51386569-db7fb900-1b5d-11e9-83b1-b9c78e7654df.png">

2. Data augmentation (flipping)

<img width="734" alt="2019-01-18 8 17 00" src="https://user-images.githubusercontent.com/40629085/51386683-47622180-1b5e-11e9-8c71-f933dcb23dea.png">

## Result
This modified GCN with ASPP model can achieve up to 80% IoU score. Belows are some of the outputs predicted by the trained model. 

Left - original image, Right - predicted mask (Salt in yellow region)

<img width="224" alt="2019-01-18 8 23 38" src="https://user-images.githubusercontent.com/40629085/51386989-5dbcad00-1b5f-11e9-9d01-ac8080242dfa.png">
