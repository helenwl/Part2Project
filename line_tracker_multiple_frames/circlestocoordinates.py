from pixelstocircles import *


#in pixelstocircles.py there is a for loop in which we record down all the X and Y values of the vacancies
#here we are putting them into an array so that we might plot the vacancies in terms of coordinates
#cXlist= cXlist[::-1]
#cYlist= cYlist[::-1]
vacancy_coordinates = {'X':cXlist, 'Y':cYlist}

#trying a pandas data frame here to plot
vacancy_coordinates_pd_array = pd.DataFrame(vacancy_coordinates, columns=['X', 'Y'])
print(vacancy_coordinates_pd_array.shape)

#trying a numpy array here to plot
vacancy_coordinates_np_array=np.array(vacancy_coordinates_pd_array)
#print(vacancy_coordinates_np_array)

#this code plots the coordinates of the vacancies. It should be the same shape as the vacancy mask for the corresponding image
'''
fig1= plt.figure()
ax= fig1.add_subplot(111)
ax.scatter(vacancy_coordinates_np_array[:,0] , vacancy_coordinates_np_array[:,1], c='b', marker='o', s=3)
#plt.xlim([0, 4092])
#these limits are because the y-coordinates would otherwise create an upside down plot
plt.ylim([4290, 0])
plt.title(f'\n{file}')
plt.show()
'''
