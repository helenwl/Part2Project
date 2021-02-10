from linehoughtransform import *
import math 
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import hdbscan
import matplotlib.image as mpimg
np.random.seed = 42

#code below tried from https://www.geeksforgeeks.org/perpendicular-distance-between-a-point-and-a-line-in-2-d/
  
# Function to find distance 
def shortest_distance(x1, y1, a, b, c, line_points, threshold_distance):  
       
    d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b)) 
    if d < threshold_distance:
        coords= tuple([x1, y1])
        line_points.append(coords)
    return line_points 

#find all the points in the vacancy_coordinates_np_array closest to line 0 (i.e. first line)
def closest_array_points(n_clusters, threshold_distance):
    results=[]
    for i in range(n_clusters):
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

        #print('line', i, 'has', len(line_points), 'points')
        #converts line_points which is currently a tuple into an array for consistency of data structures 
        line_points =np.asarray(line_points)
        #print(line_points)
        results.append(line_points)
    #np.hstack(results)
    return results

#threshold distance is the measure of the minimum perpendicular distance a point must be from a line in order for it to be counted
threshold_distance= 80

#need to make sure that line numbers and lines are consistent. Choose a line number to track and
#..and save it's equation. Then, only chose the 'line' value within a threshold distance of the iteration before
#.... in order to keep 'tracking' the same line. 
line_number=0
#n_clusters is described in the previous linehoughtransform.py script
#extracting desired line from the list of coordinates of lines 
all_line_points= closest_array_points(n_clusters, threshold_distance)
line_np_array= all_line_points[line_number]


#CODE TO KICK OUT OUTLIERS
threshold_point_distance= 600
break_point = []
line_np_array = line_np_array[np.argsort(line_np_array[:, 0])]
#print('line_np_array:', line_np_array)
for i in range(len(line_np_array)):
        x1=line_np_array[i-1,0]
        x2=line_np_array[i,0]
        y1=line_np_array[i-1,1]
        y2=line_np_array[i,1]
        actual_distance= math.sqrt((x2-x1)**2+(y2-y1)**2)

        if actual_distance > threshold_point_distance:
            break_point.append(i)
#print('break_point:', break_point)

list_of_clusters=[]
if break_point != []:
    for i in range(len(break_point)):
        #print(i)
        if i == 0:
            list_of_clusters.append(line_np_array[0:break_point[i],:])
            if len(break_point)==1:
                list_of_clusters.append(line_np_array[break_point[i]:,:])
        elif i != len(break_point)-1:
            list_of_clusters.append(line_np_array[break_point[i]:break_point[i+1],:])
        else:
            list_of_clusters.append(line_np_array[break_point[i-1]:break_point[i],:])
            list_of_clusters.append(line_np_array[break_point[i]:,:])
    #print(list_of_clusters)      
    line_np_array=max(list_of_clusters, key=len)

#number of vacancies
length = len(line_np_array)
#finding the length of the defect line
line_x_values= line_np_array[:,0]
line_y_values= line_np_array[:,1]
max_x= np.argmax(line_x_values)
min_x= np.argmin(line_x_values)

#print(max_x, min_x)
x1= line_np_array[min_x,0]
y1= line_np_array[min_x,1]
x2= line_np_array[max_x,0]
y2= line_np_array[max_x,1]
length_line= math.sqrt((x2-x1)**2 + (y2-y1)**2)
length_line=float(round(length_line,2))
min_x1 = x1
max_x2 = x2

#Visualising the chosen points

for i in range(len(unique_lines)):

    gradient=unique_lines[i,0]
    c=unique_lines[i,1]
    x = np.linspace(0,4092,100)
    y = gradient*x+c
    plt.plot(x, y,  label= f'line{i}')

plt.title(f'Points in the vacancy coordinates array close to line {line_number}. \n Number of vacancies:{length}. \n Length of line (in pixels):{length_line}. \n Frame number: {file}')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim([4290, -150])
plt.xlim([-150,4092])
plt.legend(loc='upper left')
plt.grid()

plt.scatter(line_np_array[:,0] , line_np_array[:,1], c='r', marker='o', s=3)
#plt.xlim([0, 4092])
#plt.show()

img = cv2.imread(img_dir + f'{file}.png')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23))
dilation = cv2.dilate(img,kernel,iterations = 1)
    
img = cv2.threshold(dilation, 10, 255, cv2.THRESH_BINARY)[1]
plt.imshow(img)
#plt.savefig(results_dir + f'{file}.png')
plt.show()



    