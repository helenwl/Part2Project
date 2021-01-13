from linehoughtransform import *
import math 

#code below tried from https://www.geeksforgeeks.org/perpendicular-distance-between-a-point-and-a-line-in-2-d/
  
# Function to find distance 
def shortest_distance(x1, y1, a, b, c, line_points, threshold_distance):  
       
    d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b)) 
    line_x_points=[]
    line_y_points=[]
      
    if d < threshold_distance:
        coords= tuple([x1, y1])
        line_points.append(coords)
    '''
        #line_x_points.append(x1)
        #line_y_points.append(y1)
    
    line_x_points_np_array =np.array(line_x_points)
    line_y_points_np_array = np.array(line_y_points)
    line_x_points_np_array= np.reshape(line_x_points_np_array,(len(line_x_points_np_array),1))
    line_y_points_np_array= np.reshape(line_y_points_np_array,(len(line_y_points_np_array),1))
    line_points= np.append(line_x_points_np_array, line_y_points_np_array, axis=1)
    '''
    return line_points 
#in my prototype method, the example image I am using to develop this method has 5 lines

#find all the points in the vacancy_coordinates_np_array closest to line 0 (i.e. first line)
#so far the code is counting all the vacancies in an n-SVL: it is not distinguishing between lines yet
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
        print('line', i, 'has', len(line_points), 'points')
        #converts line_points which is currently a tuple into an array for consistency of data structures 
        line_points =np.asarray(line_points)
        #print(line_points)
        results.append(line_points)
    #np.hstack(results)
    return results

#threshold distance is the measure of the minimum perpendicular distance a point must be from a line in order for it to be counted
threshold_distance= 50
line_number=0
#n_clusters is described in the previous linehoughtransform.py script
#extracting desired line from the list of coordinates of lines 
all_line_points= closest_array_points(n_clusters, threshold_distance)
line_np_array= all_line_points[line_number]
#length is the number of defects in this line
length =len(line_np_array)

#finding the length of the defect line
line_x_values= line_np_array[:,0]
line_y_values= line_np_array[:,1]
max_x= np.argmax(line_x_values)
min_x= np.argmin(line_x_values)

print(max_x, min_x)
x1= line_np_array[min_x,0]
y1= line_np_array[min_x,1]
x2= line_np_array[max_x,0]
y2= line_np_array[max_x,1]
length_line= math.sqrt((x2-x1)**2 + (y2-y1)**2)

#Visualising the chosen points
for i in range(len(unique_lines)):

    gradient=unique_lines[i,0]
    c=unique_lines[i,1]
    x = np.linspace(0,4092,100)
    y = gradient*x+c
    plt.plot(x, y,  label= f'line{i}')

plt.title(f'Points in the vacancy coordinates array close to line {line_number}. \n Number of vacancies:{length}. \n Length of line (in pixels):{length_line}')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim([4290, -150])
plt.xlim([-150,4092])
plt.legend(loc='upper left')
plt.grid()

plt.scatter(line_np_array[:,0] , line_np_array[:,1], c='b', marker='o', s=3)
#plt.xlim([0, 4092])
plt.show()