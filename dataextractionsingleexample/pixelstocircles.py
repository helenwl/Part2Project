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
path ='/home/helen/Documents/HelenExperimentalCode/Image_generation_MULTEM_largetry/predictions/Brookespredictionseverythingtogetherfolder/vacancy_mask/Cycled_004_Hour_00_Minute_00_Second_41_Frame_0003.png'
img_dir= '/home/helen/Documents/HelenExperimentalCode/Image_generation_MULTEM_largetry/predictions/linetests/'
gray = cv2.imread(path) 

#cv2.imshow('dilates',gray)
#cv2.waitKey(0)

hsv = cv2.cvtColor(gray, cv2.COLOR_BGR2HSV)

#the command below is saying what colours (RGB covention) are considered inRange
#code below turns a red vacancy map into a white vacancy map
mask = cv2.inRange(hsv, (0,200,200),(10,255,255))
#cv2.imshow('dilates',mask)
#cv2.waitKey(0)
#image saved below is raw pixels- not nice circles yet
#cv2.imwrite(os.path.join(img_dir , "white_S.png"), mask)
cv2.imwrite(img_dir + "white_S.png", mask)

#check b/w dots# threshold : converts grayscale image to binary image
th, threshed = cv2.threshold(mask, 1, 255,
                             cv2.THRESH_BINARY|cv2.THRESH_OTSU) 

# findcontours (used to find centroids of all the pixel clusters in the image)
cnts = cv2.findContours(threshed, cv2.RETR_LIST, 
                        cv2.CHAIN_APPROX_SIMPLE)[-2] 

# filter by area 
s1 = 0
s2 = 100000
xcnts = [] 

for cnt in cnts: 
    if s1<cv2.contourArea(cnt) <s2:
        xcnts.append(cnt) 
    
# printing output 
print("\nDots number: {}".format(len(xcnts))) 
#print(xcnts)

bg = np.zeros(mask.shape, np.uint8)

#Make the orig img look nicer by only having centroids
# loop over the contours
cXlist =[]
cYlist=[]
image = bg.copy()
for c in cnts:
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
cv2.imwrite(img_dir + "nice_not_dilated_S.png", image)
#now lets try to dilate the image
img = image.copy()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23))
dilation = cv2.dilate(img,kernel,iterations = 3)
    
img = cv2.threshold(dilation, 10, 255, cv2.THRESH_BINARY)[1]

cv2.imwrite(img_dir + "nice_dilated_S.png", img)

#below is trying to denoise the image to get rid of vacancies isolated
'''
dst = cv2.fastNlMeansDenoisingColored(hsv,None,10,10,7,21)

plt.subplot(121),plt.imshow(hsv)
plt.subplot(122),plt.imshow(dst)
plt.show()

img = hsv
img_bw = 255*(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 5).astype('uint8')

se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)

mask = np.dstack([mask, mask, mask]) / 255
out = img * mask

cv2.imshow('Output', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('output.png', out)
'''
# end of denoise attempt
