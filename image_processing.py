# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:12:55 2019

@author: ahall
"""

#TODO - verify the corner detection is working well (see the link at relevant section)
#corner detection needs more consideration in general, it knocks out way too many images at present - may want to add edge detection as well.
import numpy as np
import cv2
import os
import re

#rescale size
IMAGE_SIZE = 224
WHITE_THRESHOLD=0.3

#strength on which to define a corner
CORNER_THRESHOLD=0.001
#max number of detected corners allowed in image
CORNER_NUMBER=10

#crops off edges to try and get rid off the images with poor outline forms
def centre_crop(img_file,percentage):
    
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_x = np.shape(img)[0]
    img_y = np.shape(img)[1]
    
    bottom =int(img_y*percentage)
    top=int(img_y-(percentage*img_y))
    left =int(img_x*percentage)
    right=int(img_x-(percentage*img_x))
    
    cropped_img=img[left:right,bottom:top]
    return(cropped_img)
    

#this has the problem of stretching images
def resize(img,size):
    resized_image = cv2.resize(img, (size, size)) 
    return resized_image

#convert to grayscale and make BG white
def white_BG(img):
    #TODO - THE PROBLEM SEEMS TO BE HERE
    img[np.where((cv2.inRange(img, 0, 5)))] = [255]
    return img



#count the number of whit pixels - allows a rought discard of images without an outline
def count_white(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    n_white_px = np.sum(img >= 250)
    return n_white_px

#corner detection - aims to throw out images of angular blocks
#verify this on a few images: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
def count_corners(img,corner_threshold):
    dst = cv2.cornerHarris(img,10,3,0.04)
    corner_count = np.sum(dst >= corner_threshold)
    return corner_count


#uses above functions to process all the source images and write to disk
def process_all_images(source_directory,dest):
    i=0
    for root, dirs, files in os.walk(source_directory):
        for image in files:
            print(i)
            
            try:
                processed_img = centre_crop(os.path.join(root,image),0.05)
                processed_img=resize(processed_img,IMAGE_SIZE)
                processed_img = white_BG(processed_img)
                #only write images with white pixels above threshold
                if(count_white(processed_img)/IMAGE_SIZE**2>=WHITE_THRESHOLD 
                   #and count_corners(processed_img,CORNER_THRESHOLD) < CORNER_NUMBER
                   ):
                    image=re.sub("jp2","jpg",image)
                    cv2.imwrite(os.path.join(dest,image),processed_img )
            except:
                continue
    
            i=i+1
        
        
        
    
            

#%%
#scratch pad

process_all_images('S:\\Collections\\JISC_Image_Store\\Thumbnails','C:\\Users\\ahall\\Documents\\projects\\fossil hackathon\\processed')

"""processed_img = white_BG(img)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()"""
    