# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 10:59:28 2019

@author: ahall
"""
#this script sorts the processed images into named folders according to the variable we are classifying on
#JP2 JPG discrepancy

import pandas as pd
import os
import numpy as np
import shutil
import math
import random

#SOURCE_PATH = os.path.dirname(os.path.abspath(__file__))
SOURCE_PATH = "C:\\Users\\ahall\\Documents\\projects\\fossil hackathon"
SOURCE_FOLDER = "processed" #PICK A FOLDER OF PROCESSED IMAGES
CLASSIFIER_VARIABLE = "COMMON_NAME1"
TRAIN_FRACTION = 0.6


# source is the current directory
# Open dataset file
ref_file_path = "reference_table/class_labels.csv"
dataset = pd.read_csv(os.path.join(SOURCE_PATH,ref_file_path), encoding = "ISO-8859-1")
dataset = dataset[dataset[CLASSIFIER_VARIABLE].notnull()]
dataset = dataset[dataset['FILE_PATH'].notnull()]
file_names = list(dataset['FILE_PATH'].values)

img_labels = list(dataset[CLASSIFIER_VARIABLE].values)

folders_to_be_created = list(set(img_labels))

dest_train = os.path.join(SOURCE_PATH,"labelled_dataset",'train')
dest_validate = os.path.join(SOURCE_PATH,"labelled_dataset",'validate')
dest_test = os.path.join(SOURCE_PATH,"labelled_dataset",'test')

for new_path in folders_to_be_created:
    if type(new_path)==str:
        if not os.path.exists(os.path.join(dest_train ,new_path)):
            os.makedirs(os.path.join(dest_train ,new_path))
        if not os.path.exists(os.path.join(dest_validate ,new_path)):
            os.makedirs(os.path.join(dest_validate ,new_path))
        if not os.path.exists(os.path.join(dest_test ,new_path)):
            os.makedirs(os.path.join(dest_test ,new_path))


#folders = folders_to_be_created.copy()

for f in range(len(file_names)):

    current_img = file_names[f]
    current_label = img_labels[f]

    rand =  random.uniform(0, 1)
  
    if rand <= TRAIN_FRACTION:
        DEST_FOLDER = "train"
      
    else:
      
        rand2 =random.uniform(0, 1)
        if (rand2<= 0.5):
            DEST_FOLDER = 'validate'
        else:
            DEST_FOLDER = 'test'
  
    if(os.path.exists(os.path.join(SOURCE_PATH,SOURCE_FOLDER,current_img))):
        try:
            shutil.copy(os.path.join(SOURCE_PATH,SOURCE_FOLDER,current_img), os.path.join(SOURCE_PATH,"labelled_dataset",DEST_FOLDER,current_label))
      
        except:
            continue

