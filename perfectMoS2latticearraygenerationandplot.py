import numpy as np
import math
import pandas as pd
from operator import add
import random
import decimal

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#this code creates a perfect MoS2 lattice (in 2 files- one with Mo coordinates, the other with S coordinates.)
#the 2 files are then plotted in 2D (so atom height is  not visible) on top of each other.
#the files are then merged into one csv file (ready for the example_MULTEM_HRTEM.m file to read)


#this clear zone both layers
a = 3.176
b = 5.5
c = 6.1

#not sure where Brooke got these coordinates from- need to check as the hexagon height looks out of proportion
Mo1 = [0,3.677,3.05]
Mo2 = [1.588,0.917,3.05]
S1 = [0,0,0]
S2 = [0,0,6.1]
S3 = [1.588,2.75,0]
S4 = [1.588,2.75,6.1]

#this function gives the basic framework of allowed positions of atoms on the MoS2 lattice.
def basis_lattice_funt(n,m,l):
    #range(n) gives all integer values from 0 to n. the command list lists them out in an array
    x=list(range(n)) 
    #panda series are a 1D array which can hold data. It tells you the position on the array and the value corresponding to that position.
    s = pd.Series(x) 
    #turns the array into units which are in terms of the lattice constant in the a direction (one of the axes)
    x=s*a
    y=list(range(m))
    s = pd.Series(y)
    y=s*b
    z=list(range(l))
    #z should be non varying (hence why it's range is 1) because we are considering 2D MoS2- so only one layer thick
    return [[j,k,l] for j in x for k in y for l in z]

#this is making coordinate an array of positions (each new position is appended onto the end of the array)    
def coordinate(position, lattice, coordinates):   
    for i in range(len(position)): 
        for point in range(len(lattice)):
            coordinates.append(list( map(add,lattice[point], position[i]))) 

#the round function just returns the answer to a specified number of decimal places. In this case, 3 decimals
def roundCoordinates(c):
    coordinates = c
    for i in range(len(coordinates)):
        for j in range(len(coordinates[i])):
            coordinates[i][j] = round(coordinates[i][j], 3)
    return coordinates

#n = x, m = y
count=500
for i in range(1500-count):
    #n=random.randint(20,50)
    #m=random.randint(20,50)
    m=19
    n=int(m*b/a)
    l=1
    basis_lattice = basis_lattice_funt(n,m,l)
    Mo_coordinate = []
    S_coordinate = []


Mo_position = [Mo1]+[Mo2]
S_position = [S1]+[S2]+[S3]+[S4]

#generate the perfect lattice
coordinate(Mo_position, basis_lattice,Mo_coordinate)
coordinate(S_position, basis_lattice,S_coordinate)
Mo_coordinate = np.array(Mo_coordinate)
S_coordinate = roundCoordinates(S_coordinate)
S_coordinate = np.array(S_coordinate)
S_copy = S_coordinate

#now create the txt or csv file in the format required by MULTEM

Molybdenum = pd.DataFrame(Mo_coordinate)
Molybdenum.insert(3, '3', 0.05)
Molybdenum.insert(4, '4', 1.0)
Molybdenum.insert(5, '5', 0)

#MULTEM requires that the first column describes the Z number (so we know what atom it is)
Molybdenum.insert(0, 'Element', 42)


Sulphur = pd.DataFrame(S_coordinate)
Sulphur.insert(3, '3', 0.05)
Sulphur.insert(4, '4', 1.0)
Sulphur.insert(5, '5', 0)
Sulphur.insert(0, 'Element', 16)

MoS= [Molybdenum, Sulphur]

combined_file = pd.concat(MoS)
combined_file.to_csv("perfectMoS2lattice.csv")

#VISUALISATION CODE BELOW

#plotting the Sulphur coordinates and Moo coordinates onto the same graph :the 2D plot does NOT show that atoms are at different heights
#nb- plotting the scatter graphs separately means that the atoms show up different colours
plt.scatter(S_coordinate[:,0] , S_coordinate[:,1], S_coordinate[:,2])
plt.scatter(Mo_coordinate[:,0] , Mo_coordinate[:,1], Mo_coordinate[:,2])
plt.show()

#this gives a 3D plot of the above values
fig= plt.figure()
ax= Axes3D(fig)
ax.scatter(S_coordinate[:,0] , S_coordinate[:,1], S_coordinate[:,2])
ax.scatter(Mo_coordinate[:,0] , Mo_coordinate[:,1], Mo_coordinate[:,2])
plt.show()

#this plots the combined file (but Mo and S atoms don't appear as different colours in this visualisaiton)
MoSplot = np.concatenate((Mo_coordinate, S_coordinate), axis=0)
plt.scatter(MoSplot[:,0], MoSplot[:,1])
plt.show()
    
    
