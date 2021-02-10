from countpoints_closeto_houghlines import *

#find the gradients of perp lines
print('line_number:', line_number)
perp_gradient=(-1)*(1/unique_lines[line_number,0])
print('perp_gradient:', perp_gradient)
#create an array that is a selection of coordinates from line_np_array.
#... this means that perpendicular line measurements are only taken a sensible number of times along the array line
#used numpys array slicing for this which has the format of start:stop:step
sub_sampled_line_np_array= line_np_array[1::round(len(line_np_array)/20)]


#in similar fashion to countpoints_closeto_houghlines.py, I have to count the number of dots close to each drawn perpendicular line

def shortest_distance_width(x1width, y1width, a, b, c, line_points_width, threshold_distance_width):     
    d = abs((a * x1width + b * y1width + c)) / (math.sqrt(a * a + b * b)) 
    #min_x1 and max_x2 are global variables from countpoints_closeto+_houghlines.py
    
    if d < threshold_distance_width :
        coords= tuple([x1width, y1width])
        line_points_width.append(coords)
    #print('line_points_width:', line_points_width)
    #print(x1width, min_x1, max_x2)

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
            
        #print('line', i, 'has', len(line_points), 'points')
        #converts line_points which is currently a tuple into an array for consistency of data structures 
        line_points =np.asarray(line_points_width)
        #print(line_points)
        results.append(line_points_width)
    #np.hstack(results)
    return results

threshold_distance_width = 10
results_points_for_width_list_of_arrays= closest_array_points_width(threshold_distance_width)

width_of_defect_list=[]

for i in range(len(results_points_for_width_list_of_arrays)):
    width_of_defect_list.append(len(results_points_for_width_list_of_arrays[i]))
    line_width_np_array= results_points_for_width_list_of_arrays[i]
    plt.scatter(*zip(*line_width_np_array), c='b')

print('list of width measurements:', width_of_defect_list)
average_width = sum(width_of_defect_list)/len(width_of_defect_list)
max_width = max(width_of_defect_list)
print('average width', average_width)
print('max width', max_width)

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

