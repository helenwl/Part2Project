#need these 2 lines of code because my 2 python files are in different folders
import sys
sys.path.append('../perfectlatticegeneration')
#this imports the original perfect lattice code (including all the functions and variables!)
from perfectMoS2latticearraygenerationandplot import *

#this code creates 2 csv files. One is of a defective lattice, and the second csv file (named cell_holes) shows only where the defects are (i.e. a mask)

#method to make single vacancies
def V_s_vacancies(coordinate, vacancies, vac_row): 
    #random.gauss(mu, sigma) where mu= mean, sigma= standard deviation
    percent = random.gauss(1.7, 1)
    percent = abs(percent)
    #print('vacancies', percent)
    #percent = 3

    #math.ceil returns the ceiling of x, the smallest integer greater than or equal to x
    number = math.ceil(len(coordinate)*percent/100) 
    for i in range(number): 
        row = random.randint(0,len(coordinate)-i-1)
        vac_row.append(row)
        vacancies.append(coordinate[row])
        coordinate = np.delete(coordinate, row, axis=0)
        #index = np.where(coordinate[:,2]==0)
        #location = coordinate[row] 
        #index = np.where((coordinate[:,2]==0)&(coordinate[:,1]==location[1])&(coordinate[:,0]==location[0])) 
        #index = np.where((coordinate[:,1]==location[1])&(coordinate[:,0]==location[0])) 
        #print(coordinate[index])
        #coordinate = np.delete(coordinate, index, axis=0) 
    return coordinate

#method to make divacancies
def V_s2_vacancies(coordinate,vacancies): 
    percent = random.gauss(0.275, 0.09)
    percent = abs(percent)
    #print('divacancies', percent)
    number = math.ceil(len(coordinate)*percent/200) 
    for i in range(number): 
        row = random.randint(0,len(coordinate)-i-1) 
        location = coordinate[row] 
        vacancies.append(location)
        index = np.where((coordinate[:,0]==location[0])&(coordinate[:,1]==location[1])) 
        coordinate = np.delete(coordinate, index, axis=0) 
        vacancies.append(location)
    return coordinate
                    
def index_finder(X_1,Y_1,Z_1,coordinate_1):
    #round just returns the value of X_1 to a given number of decimal places
    X_1=round(X_1,3)
    Y_1=round(Y_1,3)
    # np.where returns elements chosen from x or y depending on condition. index_1 is when X,Y,Z coordinates all fit the coordinate_1 array
    index_1 = np.where((coordinate_1[:,0]==X_1)&(coordinate_1[:,1]==Y_1)&(coordinate_1[:,2]==Z_1))
    #if the shape of the array is one number.... then
    if np.array(index_1).shape == (1, 0) :
        return False
    else:
        return index_1

#this function sets X_2's value  depending on the value of direction_2. They are all S coordinates.                    
def direction_finder(X_2,Y_2,direction_2):
    if direction_2 == 0:
        X_2 +=3.176
    elif direction_2 == 1:
        X_2 -= 3.176    
    elif direction_2 == 2:
        X_2 +=1.588
        Y_2 +=2.75
    elif direction_2 == 3:
        X_2 -=1.588
        Y_2 -=2.75   
    elif direction_2 == 4:
        X_2 +=1.588
        Y_2 -=2.75       
    elif direction_2 == 5:
        X_2 -=1.588
        Y_2 +=2.75
    return X_2,Y_2

def locate(location_1):
    location_x_1 = float(location_1[0])
    location_y_1 = float(location_1[1])
    location_z_1 = float(location_1[2])
    X_1 = location_x_1
    Y_1 = location_y_1
    Z_1 = location_z_1
    return X_1, Y_1, Z_1

def second_direction(direction_3):
    if direction_3 == 0:
        options = [2,4]
    elif direction_3 == 1:
        options = [3,5]   
    elif direction_3 == 2:
        options = [0,5]
    elif direction_3 == 3:
        options = [1,4] 
    elif direction_3 == 4:
        options = [0,3]     
    elif direction_3 == 5:
        options = [1,2]
    #num is randomly picked to be either 0 or 1
    num = random.randint(0,1)
    #pick either the 1st or 2nd number of the options array with num
    new_dir = options[num]
    #then we return a single number, new_dir. This is dependent on the if clauses and the random number picked (either 0 or 1)
    return new_dir

def kink(direction_5, vacancy_list_4, coordinate_4,row_4,clear_area_4):
    #calling the second_direction function will output a number (assigned to direction_4)
    direction_4 = second_direction(direction_5)
    mu, sigma = 2, 2
    s = np.random.normal(mu, sigma, 1)
    length_4 = abs(int(s))
    #print(coordinate_4[row_4] )
    location_4 = coordinate_4[row_4]
    if len(location_4)!=0:
        location_x_4 = float(location_4[0])
        location_y_4 = float(location_4[1])
        location_z_4 = float(location_4[2])
        X_4 = location_x_4
        Y_4 = location_y_4
        Z_4 = location_z_4
        for i in range(length_4):
            X_4,Y_4 = direction_finder(X_4,Y_4,direction_4)
            find_4 = index_finder(X_4,Y_4,Z_4,coordinate_4)                
            if not isinstance(find_4, bool):
                #print('kink, find_4', find_4)
                vacancy_list_4.append(find_4)
                clear(X_4, Y_4, Z_4, direction_4, coordinate_4, clear_area_4)
                                       
def clear_append(near_dir_1, near_dir_2, X, Y, Z, coordinate, clear_area):
    X_1,Y_1 = direction_finder(X,Y,near_dir_1)
    find_1a = index_finder(X_1,Y_1,0,coordinate)
    find_1b = index_finder(X_1,Y_1,6.1,coordinate)
    X_2,Y_2 = direction_finder(X,Y,near_dir_2)
    find_2a = index_finder(X_2,Y_2,0,coordinate)
    find_2b = index_finder(X_2,Y_2,6.1,coordinate)
    if not isinstance(find_1a, bool):
        clear_area.append(find_1a[0])
    if not isinstance(find_1b, bool):
        clear_area.append(find_1b[0])
    if not isinstance(find_2b, bool):
        clear_area.append(find_2b[0])
    if not isinstance(find_2a, bool):
        clear_area.append(find_2a[0])
        
def clear(X, Y, Z, main_direction, coordinate, clear_area):
    #for i in range width
    if main_direction == 0:
        clear_append(2, 4, X, Y, 0, coordinate, clear_area)
        clear_append(2, 4, X, Y, 6.1, coordinate, clear_area)
            
    if main_direction == 1:
        clear_append(3, 5, X, Y, 0, coordinate, clear_area)
        clear_append(3, 5, X, Y, 6.1, coordinate, clear_area)
       
    if main_direction == 2:
        clear_append(0, 5, X, Y, 0, coordinate, clear_area)
        clear_append(0, 5, X, Y, 6.1, coordinate, clear_area)
        
    if main_direction == 3:
        clear_append(1, 4, X, Y, 0, coordinate, clear_area)
        clear_append(1, 4, X, Y, 6.1, coordinate, clear_area)
        
    if main_direction == 4:
        clear_append(0, 3, X, Y, 0, coordinate, clear_area)
        clear_append(0, 3, X, Y, 6.1, coordinate, clear_area)
    
    if main_direction == 5:
        clear_append(2, 1, X, Y, 0, coordinate, clear_area)
        clear_append(2, 1, X, Y, 6.1, coordinate, clear_area)
        
        
def zigzag(coordinate,vacancy_list,clear_area): 
    #percent = random.gauss(0.275, 0.09) 
    #number = math.ceil(len(coordinate)*percent/200) 
    #print('zigzag', number)
    number = 1
    #length = 5
    #direction = 0
    
    for j in range(number):
        direction = (random.randint(0,5))
        #length = random.randint(2,10)
        length = random.gauss(3, 3)
        length = abs(int(length))
        #print('This is a new vacancy along ',direction, 'of length',length)
        row = random.randint(0,len(coordinate)-1)
        
        location = coordinate[row]
        X,Y,Z = locate(location)
        #first line
        for i in range(length):
            X,Y = direction_finder(X,Y,direction)
            find = index_finder(X,Y,Z,coordinate)
                #add to clear vacancy
            if isinstance(find, bool) and (row not in clear_area):
                break
            elif isinstance(find, bool):
                break
            else:
                vacancy_list.append(find)
                #print('zigzag',find)
                clear(X, Y, Z, direction, coordinate, clear_area)

            #first bend
        if random.randint(0,10)>=4 and len(vacancy_list)>1:
            direction = second_direction(direction)            
            length = random.randint(2,10)
            row = np.array(vacancy_list[-1])
            
            if not row == False:
                location = coordinate[row[0][0]]
                X,Y,Z = locate(location)
                for i in range(length):
                    X,Y = direction_finder(X,Y,direction)
                    find = index_finder(X,Y,Z,coordinate)
                    if not isinstance(find, bool) and (row not in clear_area):
                        vacancy_list.append(find)
                        #print('zigzag_2', find)
                        clear(X, Y, Z, direction, coordinate, clear_area)
            #print(vacancy_list)
            last_row = np.array(vacancy_list[-1])
            #print(last_row)
            #last_row = np.reshape(last_row, (1, 0))
            last_row = last_row[0][0]
            #print(last_row)
            if last_row == False:
                print(last_row, 'help!')
            #further bends
            if random.randint(0,10)>=4:
                kink(direction, vacancy_list, coordinate, last_row,clear_area)
            #triple point
            if random.randint(0,10)>=4:        
                kink(direction, vacancy_list, coordinate, last_row,clear_area)
                #print('triple point!')
    return vacancy_list

#END OF FUNCTIONS--------------------------------------------------------------------------------------------------------------