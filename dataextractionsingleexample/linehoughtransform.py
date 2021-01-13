from circlestocoordinates import *
import numpy as np

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage import data
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from matplotlib import cm
import itertools

#Attempt 
#some code I am trying from https://www.learnopencv.com/hough-transform-with-opencv-c-python/#:~:text=Hough%20transform%20is%20a%20feature,lines%20etc%20in%20an%20image.&text=For%20example%2C%20a%20line%20can,x%2C%20y%2C%20r).
# Read image
#reading done in previous script
# Convert the image to gray-scale
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img=image
# Find the edges in the image using canny detector

edges = cv2.Canny(img, 50, 0)

# Detect points that form a line
threshold= 50
minLineLength= 20
maxLineGap=80
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
#print('lines',lines)
#print(lines[0])
print('threshold value:', threshold)
print('minLineLength:', minLineLength)
print('maxLineGap:', maxLineGap)
print('number of lines drawn:', len(lines))


# Draw lines on the image
#for line in lines_valid:
for line in lines:
    x1, y1, x2, y2 = line[0]
    #format is: cv2.line(image, start_point, end_point, color, thickness) 
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    

# Show result, #visualisation code
'''
#cv2.namedWindow("output", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
#img_resized = cv2.resize(img, (960, 960))                    # Resize image
#cv2.imshow("Result Image", img_resized)

plt.imshow(img)
plt.title('Hough Transform Result')
#plt.show()
'''

#extract a list of y=mx+c equations of the predicted lines
gradient_list=[]
y_intercepts=[]

for line in lines:
    x1, y1, x2, y2 = line[0]
    gradient= (y2-y1)/(x2-x1)
    y_intercept= (y1*x2-x1*y2)/(x2-x1)
    gradient_list.append(gradient)    
    y_intercepts.append(y_intercept)
 
lines_m_c_values={'X':gradient_list, 'Y':y_intercepts}   

lines_m_c_values_pd_array = pd.DataFrame(lines_m_c_values, columns=['X', 'Y'])

#convert into a numpy array here to plot
lines_m_c_values_np_array=np.array(lines_m_c_values_pd_array)

#visualisation code
'''
fig=plt.figure()
ax = fig
#plt.scatter(gradient_list, y_intercepts, s=6)
plt.scatter(lines_m_c_values_np_array[:,0], lines_m_c_values_np_array[:,1], s=6)
plt.title('Plot of m against c to show number of unique lines identified by Hough')
plt.xlabel('Gradient')
plt.ylabel('y intercept')
plt.show()
'''

#now try to cluster the coordinates to find unique lines and the coordinates of their centroids
#trying code from https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203
X= lines_m_c_values_np_array
'''
#plot elbow graph to find optimum n_clusters (it wasn't very insightful but may  help?)
wcss = []
for i in range(4, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(4, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
'''

#n_clusters indicates the number of unique lines I'd expect to find
n_clusters= 5
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)
#uncomment below to see the plots
'''
plt.scatter(X[:,0], X[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='r', s=3)
plt.title(f'Plot of m against c with red dots indicating the clustering centres. No of clusters chosen={n_clusters}')
plt.xlabel('Gradient')
plt.ylabel('y intercept')
#plt.show()
'''
unique_lines = kmeans.cluster_centers_

#this line of code sorts the array in ascending order with the gradient
unique_lines_sorted = unique_lines[np.argsort(unique_lines[:, 0])]


#code below makes all the gradients match so we don't plot lines which will criss cross
gradients_only=[]

for i in range(len(unique_lines_sorted)):
    if i !=0:
        if unique_lines_sorted[i,0]- unique_lines_sorted[i-1, 0] <= abs(0.35):
            #new= (unique_lines_sorted[i-1,0]+ unique_lines_sorted[i,0])/2
            new= unique_lines_sorted[i-1,0]
            gradients_only.append(new)
        else:
            old= unique_lines_sorted[i,0]
            gradients_only.append(old)
    else:
        first=unique_lines_sorted[i,0]
        gradients_only.append(first)
'''

for i in range(len(unique_lines_sorted)):
    if i != len(unique_lines_sorted)-1:
        if unique_lines_sorted[i,0]- unique_lines_sorted[i+1, 0] <= abs(0.35):
                new= (unique_lines_sorted[i,0]+ unique_lines_sorted[i+1,0])/2
                #new= unique_lines_sorted[i-1,0]
                gradients_only.append(new)
        else:
                old= unique_lines_sorted[i,0]
                gradients_only.append(old)
    else:
        old= unique_lines_sorted[i,0]
        gradients_only.append(old)

'''
gradients_only= np.array(gradients_only)

gradients_only= np.reshape(gradients_only,(n_clusters,1))


unique_lines_sorted = np.delete(unique_lines_sorted, 0, 1)
#
#print(gradients_only)

unique_lines_sorted= np.append(gradients_only, unique_lines_sorted, axis=1)


#uncomment/comment out this line below to either plot the gradients (with variation) or with any variation being taken out
#unique_lines = unique_lines_sorted
#print(unique_lines)
#now plot lines from kmeans.cluster_centers_

#uncomment below to see the plots
'''
for i in range(len(unique_lines)):

    gradient=unique_lines[i,0]
    c=unique_lines[i,1]
    x = np.linspace(0,4092,100)
    y = gradient*x+c
    plt.plot(x, y,  label= f'line{i}')

plt.title('Graph of lines from Hough Transform extracted using K-means clustering')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim([4290, -150])
plt.xlim([-150,4092])
plt.legend(loc='upper left')
plt.grid()

plt.scatter(vacancy_coordinates_np_array[:,0] , vacancy_coordinates_np_array[:,1], c='b', marker='o', s=3)
#plt.xlim([0, 4092])
#these limits are because the y-coordinates would otherwise create an upside down plot

#plt.show()
'''

