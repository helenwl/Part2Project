#adapted from Brooke's file of everything_together_line_tests_linear_mask.ipynb

#Lets put all our coordinates into the main pic
import cv2
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd

#import imutils
import math
#Note the predicter and splitter are on virtual computer

#read rgb mask and extract red points, convert to binary
#path = r"C:\Users\CatsP\OneDrive - Nexus365\Test subjects\day 2\concat_mask.png"
#frame_name='Cycled_004_Hour_00_Minute_00_Second_40_Frame_0005'
#path =f'/media/rob/hdd1/helen/HelenExperimentalCode/Image_generation_MULTEM_largetry/predictions/linetests/vacancy_mask_selection/{frame_name}.png'
#path =f'/media/rob/hdd1/helen/HelenExperimentalCode/Image_generation_MULTEM_largetry/predictions/vacancy_mask/Cycled_004_Hour_00_Minute_00_Second_41_Frame_0000.png'
path ='/media/rob/hdd1/helen/HelenExperimentalCode/Image_generation_MULTEM_largetry/predictions/linetests/vacancy_mask_selection/'
img_dir= '/media/rob/hdd1/helen/HelenExperimentalCode/Image_generation_MULTEM_largetry/predictions/linetests/vacancy_centroids/'
results_dir='/media/rob/hdd1/helen/HelenExperimentalCode/Image_generation_MULTEM_largetry/predictions/linetests/graph_results/'

#CHANGE batch to change the directory/set of images we are analysing
batch='mg23814-22_cycled_4_minute_00_second_39_frame_0003_to_cycled_4_minute_00_second_41_frame_0007/'
#batch='mg23814-22_cycled_4_minute_00_second_42_frame_0006_to_cycled_4_minute_00_second_47_frame_003/'
#batch='mg23814-22_cycled_6_minute_00_second_06_frame_000_to_cycled_6_minute_00_second_13_frame_003/'
#batch='mg23814-22_cycled_5_minute_1_second_29_f_1_to_minute_2_s_4_f_1/'
#batch ='mg23814-22-7_0_8_4_to_0_11_1/'
path= path + batch
img_dir= img_dir + batch
results_dir= results_dir + batch

#the relevant frames have been moved to a folder 'vacancy_selection', and then the script converts all these into centroids.
#uncomment below if running for FIRST time in NEW directory
'''
for file in os.listdir(path):
    file_image = os.path.join(path, file)

    #only uncomment this code for the first run! no need to re-calculate everytime after
    gray = cv2.imread(file_image) 
    hsv = cv2.cvtColor(gray, cv2.COLOR_BGR2HSV)
    #the command below is saying what colours (RGB covention) are considered inRange
    #code below turns a red vacancy map into a white vacancy map
    mask = cv2.inRange(hsv, (0,200,200),(10,255,255))
   
    #image saved below is raw pixels- not nice circles yet
    #cv2.imwrite(os.path.join(img_dir , "white_S.png"), mask)
    #cv2.imwrite(img_dir + "white_S.png", mask)

    #check b/w dots# threshold : converts grayscale image to binary image
    th, threshed = cv2.threshold(mask, 1, 255,
                                cv2.THRESH_BINARY|cv2.THRESH_OTSU) 

    # findcontours (used to find centroids of all the pixel clusters in the image)
    cnts = cv2.findContours(threshed, cv2.RETR_LIST, 
                            cv2.CHAIN_APPROX_SIMPLE)[-2] 

    # filter by area 
    s1 = 0
    #s2 = 100000
    s2=10000
    xcnts = [] 

    for cnt in cnts: 
        if s1<cv2.contourArea(cnt) <s2:
            xcnts.append(cnt) 
        
    # printing output 
    #print("\nDots number: {}".format(len(xcnts))) 
    #print(xcnts)

    #bg = np.zeros(mask.shape, np.uint8)
    image = np.zeros(mask.shape, np.uint8)

    #Make the orig img look nicer by only having centroids
    # loop over the contours
    cXlist =[]
    cYlist=[]
    #image = bg.copy()

    #xcnts doesn't have much meaning because we can't filter out isolated vacancies based on pixel size as they are too big.
    #for c in cnts:
    for c in xcnts:
        #https://www.learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
        # compute the coordinates of the center of the contour (axes are 4092 pixels in both x and y direction)
        M = cv2.moments(c)
        if M["m00"] ==0:
            M["m00"] = 1
        cX = int(M["m10"] /( M["m00"]))
        cY = int(M["m01"] / (M["m00"]))
        cXlist.append(cX)
        cYlist.append(cY)
        # draw the contour and center of the shape on the image
        #cv2.drawContours(image, [c], -1, (100, 255, 100), 2)
        cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)
        

        
        # show the image
    #cv2.imwrite(img_dir + f'{frame_name}.png', image)

    cv2.imwrite(img_dir + f'{file}.png', image)

'''

#CODE BELOW IS FOR RUNNING SINGLE EXAMPLES FOR TROUBLESHOOTING
#only uncomment this code for the first run! no need to re-calculate everytime after
file='Cycled_004_Hour_00_Minute_00_Second_41_Frame_0003.png'
file_path=path+file
print(file_path)
gray = cv2.imread(file_path) 
hsv = cv2.cvtColor(gray, cv2.COLOR_BGR2HSV)
#the command below is saying what colours (RGB covention) are considered inRange
#code below turns a red vacancy map into a white vacancy map
mask = cv2.inRange(hsv, (0,200,200),(10,255,255))

#image saved below is raw pixels- not nice circles yet
#cv2.imwrite(os.path.join(img_dir , "white_S.png"), mask)
#cv2.imwrite(img_dir + "white_S.png", mask)

#check b/w dots# threshold : converts grayscale image to binary image
th, threshed = cv2.threshold(mask, 1, 255,
                            cv2.THRESH_BINARY|cv2.THRESH_OTSU) 

# findcontours (used to find centroids of all the pixel clusters in the image)
cnts = cv2.findContours(threshed, cv2.RETR_LIST, 
                        cv2.CHAIN_APPROX_SIMPLE)[-2] 

# filter by area 
s1 = 0
#s2 = 100000
s2=10000
xcnts = [] 

for cnt in cnts: 
    if s1<cv2.contourArea(cnt) <s2:
        xcnts.append(cnt) 
    
# printing output 
#print("\nDots number: {}".format(len(xcnts))) 
#print(xcnts)

#bg = np.zeros(mask.shape, np.uint8)
image = np.zeros(mask.shape, np.uint8)

#Make the orig img look nicer by only having centroids
# loop over the contours
cXlist =[]
cYlist=[]
#image = bg.copy()

#xcnts doesn't have much meaning because we can't filter out isolated vacancies based on pixel size as they are too big.
#for c in cnts:
for c in xcnts:
    #https://www.learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
    # compute the coordinates of the center of the contour (axes are 4092 pixels in both x and y direction)
    M = cv2.moments(c)
    if M["m00"] ==0:
        M["m00"] = 1
    cX = int(M["m10"] /( M["m00"]))
    cY = int(M["m01"] / (M["m00"]))
    cXlist.append(cX)
    cYlist.append(cY)
    # draw the contour and center of the shape on the image
    #cv2.drawContours(image, [c], -1, (100, 255, 100), 2)
    cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)
