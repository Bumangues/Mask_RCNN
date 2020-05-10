# 6.883-Final-Project

## project

## dependencies
- [Mask_RCNN](https://github.com/matterport/Mask_RCNN)
- tensorflow 1.15
- keras 2.1.*
- pandas


### directory
```
project
|_data
  |_annotations
  |_images
  |_labels.csv
|_mask_rcnn.ipynb
|_mask_rcnn.py
|_mask_rcnn_coco.h5
```

## image_preprocessing

### dependencies 
- pillow

### run
python data_processing.py
1. Image will load with labeled boxes starting with 0
1. User is prompted to list which boxes have a human in a dangerous position
```Type skip to ignore or list which boxes are dangerous (space separated):```
1. User types 
   1.```exit``` to exit the program  
   1.```skip``` to skip image or user types a space-separated list of the labled boxes 
   1.```0 2 5``` for the boxes labeled 0, 2 and 5.
1. A new image to label is displayed 
