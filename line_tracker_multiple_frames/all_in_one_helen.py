#adapted from Brooke's file of everything_together_line_tests_linear_mask.ipynb
from configparametersforallinone import *
import cv2
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg
#import imutils
import math

path ='/media/rob/hdd1/helen/HelenExperimentalCode/Image_generation_MULTEM_largetry/predictions/linetests/vacancy_mask_selection/'
img_dir= '/media/rob/hdd1/helen/HelenExperimentalCode/Image_generation_MULTEM_largetry/predictions/linetests/vacancy_centroids/'
results_dir='/media/rob/hdd1/helen/HelenExperimentalCode/Image_generation_MULTEM_largetry/predictions/linetests/graph_results/'

path= path + batch + '/'
img_dir= img_dir + batch +'/'
#results_dir= results_dir + batch

#only for the first time, uncomment osmkdir
#writing a new directory for the new batch name
directory = f'{batch}/'
# Parent Directory path 
parent_dir = results_dir
# Path 
new_directory = os.path.join(parent_dir, directory) 
results_dir = new_directory
print(results_dir)
#os.mkdir(new_directory)

#adjustable parameters are in configrparametersforallinone.py file

'''
for i in range(n_clusters):
initial_line_number=i

directory= f'line_number_{initial_line_number}'
parent_dir= results_dir_main
new_directory = os.path.join(parent_dir, directory) 
if not os.path.exists(new_directory):
    os.makedirs(new_directory) 
results_dir= new_directory+'/'
print(results_dir)
#the relevant frames have been moved to a folder 'vacancy_selection', and then the script converts all these into centroids.
'''
#below are some variables that are stored each time the big for loop runs
#unique_lines_list_for_all_frames is to keep track of the detected lines between each file and to ensure that we track the same line
unique_lines_list_for_all_frames=[]
counter =0
#results are stored in these two lists
line_length_list=[]
number_of_vacancies_list=[]
line_defect_width_list=[]
line_defect_average_width_list=[]

#reverse value is set as True or False in configpy file
for file in sorted(os.listdir(path), reverse=reverse):

    file_image = os.path.join(path, file)
    #only uncomment this code for the first run! no need to re-calculate everytime after
    gray = cv2.imread(file_image) 
    hsv = cv2.cvtColor(gray, cv2.COLOR_BGR2HSV)
    #the command below is saying what colours (RGB covention) are considered inRange
    #code below turns a red vacancy map into a white vacancy map
    mask = cv2.inRange(hsv, (0,200,200),(10,255,255))

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

    image = np.zeros(mask.shape, np.uint8)

    #Make the orig img look nicer by only having centroids
    # loop over the contours
    cXlist =[]
    cYlist=[]

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

    #UNCOMMENT THIS when running in a NEW directory for the FIRST TIME
    cv2.imwrite(img_dir + f'{file}.png', image)

####################pixelstocirclesEND

    #in pixelstocircles.py there is a for loop in which we record down all the X and Y values of the vacancies
    #here we are putting them into an array so that we might plot the vacancies in terms of coordinates
    #cXlist= cXlist[::-1]
    #cYlist= cYlist[::-1]
    vacancy_coordinates = {'X':cXlist, 'Y':cYlist}

    #trying a pandas data frame here to plot
    vacancy_coordinates_pd_array = pd.DataFrame(vacancy_coordinates, columns=['X', 'Y'])


    #trying a numpy array here to plot
    vacancy_coordinates_np_array=np.array(vacancy_coordinates_pd_array)


    #this code plots the coordinates of the vacancies. It should be the same shape as the vacancy mask for the corresponding image
    if seegraphs==True:
        fig1= plt.figure()
        ax= fig1.add_subplot(111)
        ax.scatter(vacancy_coordinates_np_array[:,0] , vacancy_coordinates_np_array[:,1], c='b', marker='o', s=3)
        #plt.xlim([0, 4092])
        #these limits are because the y-coordinates would otherwise create an upside down plot
        plt.ylim([4290, -150])
        plt.show()
    
    #################circlestocoordinatesEND

    import numpy as np

    from skimage.transform import hough_line, hough_line_peaks
    from sklearn.cluster import MeanShift, estimate_bandwidth
    from skimage.feature import canny
    from skimage import data
    from sklearn.cluster import KMeans

    import matplotlib.pyplot as plt
    from matplotlib import cm
    import itertools
    #some code I am trying from https://www.learnopencv.com/hough-transform-with-opencv-c-python/#:~:text=Hough%20transform%20is%20a%20feature,lines%20etc%20in%20an%20image.&text=For%20example%2C%20a%20line%20can,x%2C%20y%2C%20r).


    img=image
    # Find the edges in the image using canny detector

    edges = cv2.Canny(img, 50, 0)

    # Detect points that form a line
    #n_clusters is 5 for batch 1. 2 for batch 2
    n_clusters= n_clusters
    threshold= threshold
    minLineLength= minLineLength
    maxLineGap=maxLineGap
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
    
    print('threshold value:', threshold)
    print('minLineLength:', minLineLength)
    print('maxLineGap:', maxLineGap)

    print('number of lines drawn:', len(lines))
    #self-regulation of hough transform parameters

    while len(lines) > 200:
        threshold= round(1.05*threshold)
        minLineLength= round(1.05*minLineLength)
        #maxLineGap=round(0.95*maxLineGap)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
        print('number of lines drawn:', len(lines))

    while len(lines)< 20:
        threshold= round(0.9*threshold)
        minLineLength= round(0.9*minLineLength)
        maxLineGap=round(1.1*maxLineGap)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)



    # Draw lines on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        #format is: cv2.line(image, start_point, end_point, color, thickness) 
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        

    # Show result, #visualisation code
    if seegraphs== True:
        plt.imshow(img)
        plt.title(f'Hough Transform Result \n{file}')
        plt.show()
            

    #extract a list of y=mx+c equations of the predicted lines
    gradient_list=[]
    y_intercepts=[]
    print('lines', lines.shape[0])
    keep_count=-1
    for line in lines:
        keep_count = keep_count+1
        x1, y1, x2, y2 = line[0]

        #if the lines drawn overlap directly, they will break the function!
        if x1 == x2 or y1 == y2:
            print('OH NO')
            print('length of lines', len(lines))
            print(keep_count)
            print('x1, y1,x2,y2:', x1, y1, x2, y2)
            #pretty sure the -1 is NOT RIGHT (take it out!?)
            lines = np.delete(lines, (keep_count-10), axis=0)
            
        else:
            gradient= (y2-y1)/(x2-x1)
            y_intercept= (y1*x2-x1*y2)/(x2-x1)
            gradient_list.append(gradient)    
            y_intercepts.append(y_intercept)
    
    lines_m_c_values={'X':gradient_list, 'Y':y_intercepts}   

    lines_m_c_values_pd_array = pd.DataFrame(lines_m_c_values, columns=['X', 'Y'])

    #convert into a numpy array here to plot
    lines_m_c_values_np_array=np.array(lines_m_c_values_pd_array)

    #visualisation code
    if seegraphs== True:
        fig=plt.figure()
        ax = fig
        #plt.scatter(gradient_list, y_intercepts, s=6)
        plt.scatter(lines_m_c_values_np_array[:,0], lines_m_c_values_np_array[:,1], s=6)
        plt.title('Plot of m against c to show number of unique lines identified by Hough')
        plt.xlabel('Gradient')
        plt.ylabel('y intercept')
        plt.show()
    

    #now try to cluster the coordinates to find unique lines and the coordinates of their centroids
    #trying code from https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203
    X= lines_m_c_values_np_array
    '''
    #plot elbow graph to find optimum n_clusters (it wasn't very insightful but may  help?)
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(4, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    '''
    #uncomment for k-mean algo code
    
    #n_clusters indicates the number of unique lines I'd expect to find
    n_clusters= n_clusters
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(X)
    #uncomment below to see the plots
    if seegraphs== True:
        plt.scatter(X[:,0], X[:,1])
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='r', s=3)
        plt.title(f'Plot of m against c with red dots indicating the clustering centres. \n No of clusters chosen={n_clusters}')
        plt.xlabel('Gradient')
        plt.ylabel('y intercept')
        plt.show()
    
    unique_lines = kmeans.cluster_centers_
    
    #mean-shift clustering algo code
    '''
    bandwidth = estimate_bandwidth(X, quantile=0.3, n_samples=700)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    unique_lines =  ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)

    # Plot result
    import matplotlib.pyplot as plt
    from itertools import cycle

    plt.figure(1)
    plt.clf()

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    n_clusters=n_clusters_
    '''
    ######
    #print('unique lines:', unique_lines)
    unique_lines = unique_lines[np.argsort(unique_lines[:, 0])]
    #ask the user to choose which line they want to track
    
    unique_lines =np.asarray(unique_lines)
    
    
    print('unique lines:',unique_lines)
    #now plot lines from kmeans.cluster_centers_

    #uncomment below to see the plots
    #this plot shows ALL the predicted lines
    if seegraphs== True:
        for i in range(len(unique_lines)):

            gradient=unique_lines[i,0]
            c=unique_lines[i,1]
            x = np.linspace(0,4092,100)
            y = gradient*x+c
            plt.plot(x, y,  label= f'line{i}')
        
        plt.title(f'Graph of lines from Hough Transform extracted using K-means clustering \n{file}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim([4290, -150])
        plt.xlim([-150,4092])
        plt.legend(loc='upper left')
        plt.grid()

        plt.scatter(vacancy_coordinates_np_array[:,0] , vacancy_coordinates_np_array[:,1], c='b', marker='o', s=3)
        #plt.xlim([0, 4092])
        #these limits are because the y-coordinates would otherwise create an upside down plot
        plt.show()
    
    ###########linehoughtransformsEND

    import math 

    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    import hdbscan
    np.random.seed = 42

    #code below tried from https://www.geeksforgeeks.org/perpendicular-distance-between-a-point-and-a-line-in-2-d/
    
    # Function to find distance 
    def shortest_distance(x1, y1, a, b, c, line_points, threshold_distance):  
        
        d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b)) 

        if d < threshold_distance:
            coords= tuple([x1, y1])
            line_points.append(coords)
        return line_points 
    #in my prototype method, the example image I am using to develop this method has 5 lines

    #find all the points in the vacancy_coordinates_np_array closest to line 0 (i.e. first line)
    #so far the code is counting all the vacancies in an n-SVL: it is not distinguishing between lines yet
    def closest_array_points(length_of_unique_lines, threshold_distance):
        results=[]
        for i in range(length_of_unique_lines):
            line_number = i
            line_points=[]
            
            #a, b, c are not the same as m and c because the equation has been rearranged
            a= - unique_lines[line_number, 0]
            b = 1
            c= - unique_lines[line_number,1]

            for j in range(len(vacancy_coordinates_np_array)):
                x1 = vacancy_coordinates_np_array[j,0]
                y1 = vacancy_coordinates_np_array[j,1]
                line_points= (shortest_distance(x1, y1, a, b, c, line_points, threshold_distance))
            #converts line_points which is currently a tuple into an array for consistency of data structures 
            line_points =np.asarray(line_points)
            results.append(line_points)
        return results

    #threshold distance is the measure of the minimum perpendicular distance a point must be from a line in order for it to be counted
    threshold_distance= threshold_distance

    #the line_number of initially labelled image that we'd like to track.

    #this value of line_number is ONLY valid for the 1st run when the user has to input what line they want to track
    if counter == 0:
        line_number=initial_line_number

    if seegraphs== True:  
        for i in range(len(unique_lines)):
            gradient=unique_lines[i,0]
            c=unique_lines[i,1]
            x = np.linspace(0,4092,100)
            y = gradient*x+c
            plt.plot(x, y,  label= f'line{i}')

        plt.title(f'Graph of lines from Hough Transform extracted using K-means clustering \n{file}. \n this is the first file in the series. \n you need to choose the number of the line you choose to track. \n line currently selected is: line {line_number}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim([4290, -150])
        plt.xlim([-150,4092])
        plt.legend(loc='upper left')
        plt.grid()

        plt.scatter(vacancy_coordinates_np_array[:,0] , vacancy_coordinates_np_array[:,1], c='b', marker='o', s=3)
        #plt.xlim([0, 4092])
        #these limits are because the y-coordinates would otherwise create an upside down plot

        plt.show()
           

    #this code below tracks the same line by comparing lines with previous iteration to make sure lines dont jump
    #if this is not the first iteration:
    if counter != 0:
        comparison = unique_lines_list_for_all_frames[counter-1]
        line_to_stick_with = comparison[line_number]
        print('comparison:', comparison)
        print('line to stick with:', line_to_stick_with)
        m1 = line_to_stick_with[0]
        c1 = line_to_stick_with[1]
        alike_lines_metric=[]
        for k in range(len(unique_lines)):
            m2 = unique_lines[k,0]
            c2= unique_lines[k,1]
            #alike_lines_metric should be a number as close to 0 as possible to indicate that the lines are similar
            #alike_lines_metric.append(abs(((m2/m1)+(c2/c1))/2 -1))
            #alike_lines_metric.append(abs((m2/m1)*(c1/c2)-1))
            alike_lines_metric.append(abs(m2/m1 -1)+ abs(c2/c1 -1))
        print('min alike lines:', min(alike_lines_metric))

        #the lines have to be at least a little alike before it is accepted
        if min(alike_lines_metric)< alike_lines_tolerance:
            line_number_new = alike_lines_metric.index(min(alike_lines_metric))
            print('line_number_new', line_number_new)
            line_number = line_number_new
            #print('line_number:', line_number)
        else:
            print('need to change!')
            alike_lines_metric=[]
            for i in range(len(X)):
                m2= X[i,0]
                c2= X[i,1]
                alike_lines_metric.append(abs(m2/m1 -1)+ abs(c2/c1 -1))
            min_index = alike_lines_metric.index(min(alike_lines_metric))
            print(X[min_index,0], X[min_index,1])
            print('alike_lines_metric:', alike_lines_metric)
            print('min index:', min_index)
            print('min alike lines:', min(alike_lines_metric))
            if min(alike_lines_metric) < alike_lines_tolerance:
                print('Hough Line will fit instead')
                line_equation_m_new = X[min_index,0]
                line_equation_c_new = X[min_index,1]
                line_equation_new = np.array([(line_equation_m_new), (line_equation_c_new)])
                line_equation_new = line_equation_new.reshape(1,2)
                print(line_equation_new)
                
                unique_lines = np.concatenate((unique_lines, line_equation_new))
            #if no hough transform detected line fits, then stick to the old line
            else:
                print('no Hough line- need to stick to old line')
                line_to_stick_with= line_to_stick_with.reshape(1,2)
                unique_lines = np.concatenate((unique_lines, line_to_stick_with))

            #print('unique lines post append', unique_lines)
            #lenght - 1 because of how indexing work (it starts from 0)
            line_number = len(unique_lines)-1
            print('new line number:', line_number)
    #n_clusters is described in the previous linehoughtransform.py script
    #extracting desired line from the list of coordinates of lines 
    #all_line_points represents the 'result' output from the closests_array_points function
    print('unique lines:', unique_lines)
    length_of_unique_lines = len(unique_lines)
    all_line_points= closest_array_points(length_of_unique_lines, threshold_distance)

    line_np_array= all_line_points[line_number]

    #CODE TO KICK OUT OUTLIERS
    threshold_point_distance= threshold_point_distance
    break_point = []
    #:0 means sort by x value. :1 means sort by y value. this is for picking outliers
    if shallow ==True:
        line_np_array = line_np_array[np.argsort(line_np_array[:, 0])]
    else:
        line_np_array = line_np_array[np.argsort(line_np_array[:, 1])]

    for i in range(len(line_np_array)):
        if i!= 0:
            x1=line_np_array[i-1,0]
            x2=line_np_array[i,0]
            y1=line_np_array[i-1,1]
            y2=line_np_array[i,1]
            actual_distance= math.sqrt((x2-x1)**2+(y2-y1)**2)

            if actual_distance > threshold_point_distance:
                break_point.append(i)
  
    list_of_clusters=[]
    if break_point != []:
        for i in range(len(break_point)):
            if i == 0:
                list_of_clusters.append(line_np_array[0:break_point[i],:])
                if len(break_point)==1:
                    list_of_clusters.append(line_np_array[break_point[i]:,:])
            #len(break_point at 3 is a special case) and len(break_point) != 3
            elif i != len(break_point)-1:
                list_of_clusters.append(line_np_array[break_point[i-1]:break_point[i],:])
            else:
                list_of_clusters.append(line_np_array[break_point[i-1]:break_point[i],:])
                list_of_clusters.append(line_np_array[break_point[i]:,:])
        print('break_point', break_point)
        print(list_of_clusters)      
        line_np_array=max(list_of_clusters, key=len)

    #length is the number of defects in this line
    length = np.count_nonzero(line_np_array)/2

    #finding the length of the defect line
    line_x_values= line_np_array[:,0]
    line_y_values= line_np_array[:,1]
    max_x= np.argmax(line_x_values)
    min_x= np.argmin(line_x_values)

    x1= line_np_array[min_x,0]
    y1= line_np_array[min_x,1]
    x2= line_np_array[max_x,0]
    y2= line_np_array[max_x,1]
    length_line= math.sqrt((x2-x1)**2 + (y2-y1)**2)
    length_line=float(round(length_line,2))
    unique_lines_list_for_all_frames.append(unique_lines)
    line_length_list.append(length_line)
    number_of_vacancies_list.append(length)

    #Visualising the chosen points
    
    for i in range(len(unique_lines)):

        gradient=unique_lines[i,0]
        c=unique_lines[i,1]
        x = np.linspace(0,4092,100)
        y = gradient*x+c
        plt.plot(x, y,  label= f'line{i}')

    #plt.title(f'Points in the vacancy coordinates array close to line {line_number}. \n Number of vacancies:{length}. \n Length of line (in pixels):{length_line}. \n Frame number: {file}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim([4290, -150])
    plt.xlim([-150,4092])
    plt.legend(loc='upper left')
    plt.grid()

    plt.scatter(line_np_array[:,0] , line_np_array[:,1], c='r', marker='o', s=2)
    

    #PLOTTING RESULTS ONTO IMAGE
    img = cv2.imread(img_dir + f'{file}.png')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23))
    dilation = cv2.dilate(img,kernel,iterations = 1)
        
    img = cv2.threshold(dilation, 10, 255, cv2.THRESH_BINARY)[1]
    plt.imshow(img)
    #plt.show()
    #plt.savefig(results_dir + f'{file}.png')
    #plt.clf()
    ######## countpoints_closeto_lineEND
    
    ##start of width_measuring function

    #find the gradients of perp lines
    perp_gradient=(-1)*(1/unique_lines[line_number,0])
    #create an array that is a selection of coordinates from line_np_array.
    #... this means that perpendicular line measurements are only taken a sensible number of times along the array line
    #used numpys array slicing for this which has the format of start:stop:step
    #use /0.3*len(linearray) to make sure that the number we are diviiding by is normalised to the sample
    sub_sampled_line_np_array= line_np_array[1::round(len(line_np_array)/0.3*len(line_np_array))]


    #in similar fashion to countpoints_closeto_houghlines.py, I have to count the number of dots close to each drawn perpendicular line

    def shortest_distance_width(x1width, y1width, a, b, c, line_points_width, threshold_distance_width):     
        d = abs((a * x1width + b * y1width + c)) / (math.sqrt(a * a + b * b)) 
        #min_x1 and max_x2 are global variables from countpoints_closeto+_houghlines.py
        
        if d < threshold_distance_width :
            coords= tuple([x1width, y1width])
            line_points_width.append(coords)

        return line_points_width

    def closest_array_points_width(threshold_distance_width):
        results=[]
        for i in range(len(sub_sampled_line_np_array)):
            line_number = i
            #a, b, c are not the same as m and c because the equation has been rearranged
            line_points_width=[]
            b=1
            a= -perp_gradient
            c= -(sub_sampled_line_np_array[i,1] + a* sub_sampled_line_np_array[i,0])
            for j in range(len(line_np_array)):
                x1 =line_np_array[j,0]
                y1 = line_np_array[j,1]
                
                line_points= (shortest_distance_width(x1, y1, a, b, c, line_points_width, threshold_distance_width))

            #converts line_points which is currently a tuple into an array for consistency of data structures 
            line_points =np.asarray(line_points_width)
           
            results.append(line_points_width)
        return results

    threshold_distance_width = threshold_distance_width
    results_points_for_width_list_of_arrays= closest_array_points_width(threshold_distance_width)

    width_of_defect_list=[]

    for i in range(len(results_points_for_width_list_of_arrays)):
        width_of_defect_list.append(len(results_points_for_width_list_of_arrays[i]))
        line_width_np_array= results_points_for_width_list_of_arrays[i]
        #plt.scatter(*zip(*line_width_np_array), c='b')

    average_width = sum(width_of_defect_list)/len(width_of_defect_list)
    max_width = max(width_of_defect_list)

    line_defect_width_list.append(max_width)
    line_defect_average_width_list.append(average_width)

    if seegraphs== True:
        for i in range(len(sub_sampled_line_np_array)):
            #c is y1 + x1/m1 (see notebook for simple algebra i used to get here)
            x1width = sub_sampled_line_np_array[i,1]
            y1width = sub_sampled_line_np_array[i,0]
            c= x1width - y1width*perp_gradient
            x = np.linspace(x1,x2,10000)
            y = perp_gradient*x+c
            plt.plot(x, y,  label= f'line{i}')
        plt.title(f'Points in the vacancy coordinates array close to line {line_number}. \n Number of vacancies:{length}. \n Length of line (in pixels):{length_line}. \n Frame number: {file}. \n Line width: {max_width}')

        plt.show()
    

    #plotting the graph but the title and save_img code has been moved down in order to include line width info
    plt.title(f'Points in the vacancy coordinates array close to line {line_number}. \n Number of vacancies:{length}. Length of line (in pixels):{length_line}. Line width: {max_width}\n Frame number: {file}.')
    #print(results_dir+f'{file}')
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.savefig(results_dir + f'{file}')  
    #plt.show()  
    plt.clf()

    print('file name:', file)
    counter = counter +1
    print(counter)

## Code from plotting_results_from_all_in_one_helen.py

number_of_frames= len(os.listdir(path))

x = list(range(0, number_of_frames))
x = [i * 0.125 for i in x]

if reverse== True:
    #reverse lists because we did in reverse order
    line_length_list.reverse()
    number_of_vacancies_list.reverse()
    line_defect_average_width_list.reverse()
    line_defect_width_list.reverse()

#SAVE RESULTS AS CSV WITH ALL NECESSARY PARAMETERS

results_data = pd.DataFrame()
results_data.insert(0, 'frame', sorted(os.listdir(path)))
results_data.insert(1, 'time/s', x)
results_data.insert(2, 'line length/pixels', line_length_list)
results_data.insert(3, 'number of vacancies', number_of_vacancies_list)
results_data.insert(4, 'initial line number', initial_line_number)
results_data.insert(5, 'n_clusters', n_clusters)
results_data.insert(6, 'hough threshold', threshold)
results_data.insert(7, 'hough minlinelength', minLineLength)
results_data.insert(8, 'hough maxLineGap', maxLineGap)
results_data.insert(9, 'threshold distance', threshold_distance)
results_data.insert(10, 'threshold point distance', threshold_point_distance)
results_data.insert(11, 'threshold width distance', threshold_distance_width)

if reverse == True:
    #we are reading files backwards for the sake of line detection so need to reverse the order of the csv file
    results_data.reindex(index=results_data.index[::-1])

results_data.to_csv(results_dir+ f'{batch}_{initial_line_number}.csv')

#REPEAT BUT FOR LINE LENGTH
#calculating line of best fit, assumiing linearity
m, c = np.polyfit(x, line_length_list, 1)
y_best_fit = []
for i in range(len(x)):
    y_best_fit.append(m*x[i]+c)
correlation_matrix = np.corrcoef(x, line_length_list)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
plt.plot(x, y_best_fit, label=f'Line of best fit: y={m}x+{c}. \n R value: {r_squared}')
plt.plot(x, line_length_list, label='Experimental Data')
plt.xlabel('Time/seconds')
plt.ylabel('Line Length/pixels')
plt.title(f'Line length vs Time \n tracking line number: {initial_line_number}. \n {batch}')
plt.legend(loc='upper left')
plt.savefig(results_dir + f'{batch}_length_of_line_defect.png')
plt.show()
plt.clf()

#PLOTTING FOR VACANCY NUMBER
#calculating line of best fit, assumiing linearity
m, c = np.polyfit(x, number_of_vacancies_list, 1)
y_best_fit = []
for i in range(len(x)):
    y_best_fit.append(m*x[i]+c)

correlation_matrix = np.corrcoef(x, number_of_vacancies_list)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2

plt.plot(x, y_best_fit, label=f'Line of best fit: y={m}x+{c}. \n R value: {r_squared}')
plt.plot(x, number_of_vacancies_list, label='Experimental Data')
plt.xlabel('Time/seconds')
plt.ylabel('Number of vacancies in line')
plt.title(f'No. of Vacancies vs Time \n tracking line number: {initial_line_number}. \n {batch}')
plt.legend(loc='upper left')
plt.savefig(results_dir + f'{batch}_number_of_vacancies.png')
plt.show()
plt.clf()

#PLOTTING FOR MAX LINE DEFECT WIDTH
#calculating line of best fit, assumiing linearity
m, c = np.polyfit(x, line_defect_width_list, 1)
y_best_fit = []
for i in range(len(x)):
    y_best_fit.append(m*x[i]+c)
correlation_matrix = np.corrcoef(x, line_defect_width_list)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
plt.plot(x, y_best_fit, label=f'Line of best fit: y={m}x+{c}. \n R value: {r_squared}')
plt.plot(x, line_defect_width_list, label='Experimental Data')
plt.xlabel('Time/seconds')
plt.ylabel('Max Line Width')
plt.title(f'Max Line Width vs Time \n tracking line number: {initial_line_number}. \n {batch}')
plt.legend(loc='upper left')
#plt.savefig(results_dir + f'{batch}_max_width_of_line_defect.png')
plt.show()
plt.clf()

#PLOTTING FOR AVERAGE LINE DEFECT WIDTH
#calculating line of best fit, assumiing linearity
m, c = np.polyfit(x, line_defect_average_width_list, 1)
y_best_fit = []
for i in range(len(x)):
    y_best_fit.append(m*x[i]+c)
correlation_matrix = np.corrcoef(x, line_defect_average_width_list)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
plt.plot(x, y_best_fit, label=f'Line of best fit: y={m}x+{c}. \n R value: {r_squared}')
plt.plot(x, line_defect_average_width_list, label='Experimental Data')
plt.xlabel('Time/seconds')
plt.ylabel('Average Line Width')
plt.title(f'Average Line Width vs Time \n tracking line number: {initial_line_number}. \n {batch}')
plt.legend(loc='upper left')
#plt.savefig(results_dir + f'{batch}_average_width_of_line_defect.png')
plt.show()
plt.clf()


##Code from results_video_visualiser.py
image_folder = results_dir
print('imagefolder:', image_folder)


video_name= image_folder+ f'{batch}.avi'

images = [file for file in sorted(os.listdir(image_folder)) if file.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()


