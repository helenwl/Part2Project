import numpy as np
import math
import pandas as pd
from operator import add
import random
import decimal
#this code creates a perfect MoS2 lattice (in 2 files- one with Mo coordinates, the other with S coordinates.)
#the 2 files are then plotted in 2D (so atom height is  not visible) on top of each other

#this clear zone both layers

a = 3.176
b = 5.5
c = 6.1
#not sure where Brooke got these coordinates from
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
    
def coordinate(position, lattice, coordinates):   
    for i in range(len(position)): 
        for point in range(len(lattice)):
            coordinates.append(list( map(add,lattice[point], position[i]))) 

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


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

Mo_position = [Mo1]+[Mo2]
S_position = [S1]+[S2]+[S3]+[S4]

#generate the perfect lattice
coordinate(Mo_position, basis_lattice,Mo_coordinate)
coordinate(S_position, basis_lattice,S_coordinate)
Mo_coordinate = np.array(Mo_coordinate)
S_coordinate = roundCoordinates(S_coordinate)
S_coordinate = np.array(S_coordinate)
S_copy = S_coordinate

#this saves it as a separate csv. Currently Mo and S coordinates are being saved in 2 separate csv files, but it should be the same file
#to be read successfully by MULTEM.
pd.DataFrame(S_coordinate).to_csv("S_arraytest.csv")
pd.DataFrame(Mo_coordinate).to_csv("Mo_arraytest.csv")

#plotting the Sulphur coordinates and Moo coordinates onto the same graph:the 2D plot does NOT show that atoms are at different heights
plt.scatter(S_coordinate[:,0] , S_coordinate[:,1])
plt.scatter(Mo_coordinate[:,0] , Mo_coordinate[:,1])
plt.show()
    