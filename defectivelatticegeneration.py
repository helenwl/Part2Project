

#transformations = []
#transformations = np.array(transformations)  


#To create lots of scenarios: this script is run multiple times (via a for loop)
count=0
for i in range(10-count):

    #important that the import is within the for loop so that defects are not cumulative
    from defectivelatticegenerationfunctions import *
    #Create a line defect (a row of vacancies)----------------------------------------------------------------------------------------------
    #create first vacancy in row
    #vacancy_hole describes coordinates of agglomerated vacancies which have formed a line defect

    #clear_zone prevents the overlapping of defects during generation
    clear_zone = []
    vacancy_hole = []
    clear_zone = np.array(clear_zone)  
    vacancy_hole = np.array(vacancy_hole)
    percent_lines = random.gauss(15, 5)
    #number_lines = math.ceil(len(S_copy)*percent_lines/100) 
    print(int(percent_lines))

    #this sets up the basic arrays from which we will make the list of vacancy coordinates
    #we call int(percent_lines) because we want an integer number
    for k in range(int(percent_lines)):
        vacancy_rows = []
        #clear_rows and clear_zone exist as a way to stop the unrealistic overlapping of the defects- it's a temporary array that is used during generation
        clear_rows = []
        #makes the shape of the array of clear_zone (0,3)
        clear_zone = np.reshape(clear_zone, (len(clear_zone), 3))
        #makes the shape of the array of vacancy_hole (0,3)
        vacancy_hole = np.reshape(vacancy_hole, (len(vacancy_hole), 3))

        vacancy_rows = zigzag(S_coordinate,vacancy_rows,clear_rows)

    #create the array of vacancy and clear co-ords (not rows)
        for i in range(len(clear_rows)):
            clear_zone = np.append(clear_zone, S_coordinate[(clear_rows[i])], axis=0)
            '''if i == 0:
                print('this is the beginning', clear_zone)
            elif i == len(clear_rows)-1:
                print('this is the end', clear_zone)'''
        #print(clear_zone)
        for i in range(len(vacancy_rows)):
            if (S_coordinate[vacancy_rows[i]]).shape == (1,3):
                vacancy_hole = np.append(vacancy_hole, S_coordinate[vacancy_rows[i]], axis=0)

    #remove rows from S array (which goes back into the above) in the next iteration of the for loop
        if vacancy_rows == False:
            print('I think this will fault')
        #print('remove rows', len(S_coordinate))
        #print(len(vacancy_rows))
        #print((vacancy_rows))
        S_coordinate = np.delete(S_coordinate, vacancy_rows, axis=0)
        #S_coordinate = np.delete(S_coordinate, clear_rows, axis=0)
        #print('success!', k)
        #print(k, len(vacancy_hole))

    #Create a single vacancies----------------------------------------------------------------------------------------------
    #VS_holes describes the coordinate of single sulphur vacancies
    #add individual holes?
    VS_rows = []
    VS_holes = []
        
    S_vacancies = V_s_vacancies(S_coordinate,VS_holes,VS_rows)
    #print(VS_rows)
    #print('here be holes',VS_holes,np.shape(VS_holes))

    #S2_vacancies = cell.V_s2_vacancies(S_vacancies,VS_rows)
    #for i in range(len(VS_rows)):
        #VS_holes = np.append(VS_holes, S_coordinate[VS_rows[i]], axis=0)
        #vacancy_hole = np.append(vacancy_hole, S_coordinate[VS_rows[i]], axis=0)

    #S_coordinate = np.delete(S_coordinate, VS_rows, axis=0)
    VS_holes = np.reshape(VS_holes, (len(VS_holes), 3))
    for i in range(len(VS_holes)):
        index_hole = np.where((S_copy[:,0]==VS_holes[i,0])&(S_copy[:,1]==VS_holes[i,1])&(S_copy[:,2]==VS_holes[i,2]))
        #if np.array(index_hole).shape != (1, 0) :
        S_copy = np.delete(S_copy, index_hole, axis=0)

    #remove vacancies from s_copy
    for i in range(len(vacancy_hole)):
        index_hole = np.where((S_copy[:,0]==vacancy_hole[i,0])&(S_copy[:,1]==vacancy_hole[i,1])&(S_copy[:,2]==vacancy_hole[i,2]))
        #if np.array(index_hole).shape != (1, 0) :
        S_copy = np.delete(S_copy, index_hole, axis=0)

        
    zz = S_copy

    total_holes = len(vacancy_hole)+len(VS_holes)
    total_missing = 2*len(Mo_coordinate) - len(zz)
    discrepancy = total_holes - total_missing
    #transformations = np.append(transformations, [count,total_holes,total_missing,discrepency],axis=0)


    #  '''myCsvRow=[count,total_holes,total_missing,discrepency]
    #  with open('E:\\Hilary term\\First trial\\Coordinates\\Transformations.csv','a') as fd:
    #     writer = csv.writer(fd)
    #    writer.writerow(myCsvRow)
        #   '''

    #Writing into a CSV file (for MULTEM)-------------------------------------------------------------------------------------------------------------


    Molybdenum = pd.DataFrame(Mo_coordinate, columns=['X', 'Y', 'Z'])
    Molybdenum.insert(0, 'Element', 42)

    #Sulphur = pd.DataFrame(S_vacancies, columns=['X', 'Y', 'Z'])
    #Sulphur = pd.DataFrame(S2_vacancies, columns=['X', 'Y', 'Z'])
    Sulphur = pd.DataFrame(zz, columns=['X', 'Y', 'Z'])
    Sulphur.insert(0, 'Element', 16)
    first_round = pd.concat([Molybdenum, Sulphur], ignore_index=True)
    #supercell = superround(3)

    path_location  = "C:\\Users\\helen\\iCloudDrive\\Documents\\UNI\\4th year\\Code\\latticegeneration\\defectivelatticegeneration"

    #first_round.to_csv(f'{path_location}\\defectiveMoS2lattice_{count}.csv')
    first_round.to_csv(f'defectiveMoS2lattice_{count}.csv')

    #Writing into a CSV file (for corresponding mask)-------------------------------------------------------------------------------------------------------------
    if len(vacancy_hole>0) and len(VS_holes>0):
        holes_1 = pd.DataFrame(VS_holes, columns=['X', 'Y', 'Z'])
        holes_2 = pd.DataFrame(vacancy_hole, columns=['X', 'Y', 'Z'])
        holes = pd.concat([holes_1,holes_2], ignore_index=True)

        print(holes)
        holesplot= np.array(holes)
        plt.scatter(holesplot[:,0] , holesplot[:,1], holesplot[:,2])
        plt.title([count])
        plt.show()
        
    elif len(vacancy_hole>0):
        holes = pd.DataFrame(vacancy_hole, columns=['X', 'Y', 'Z'])
        plt.scatter(holes[:,0] , holes[:,1], holes[:,2])
        plt.show()
        
    elif len(VS_holes>0):
        holes = pd.DataFrame(VS_holes, columns=['X', 'Y', 'Z'])
        print(holes)
        plt.scatter(holes[:,0] , holes[:,1], holes[:,2])
    
        plt.show()

    holes.to_csv(f'defectiveMoS2lattice_cellholes_{count}.csv')

    count+=1

#VISUALISATION CODE BELOW--------------------------------------------------------------------------------
'''
#print(len(S_coordinate),len(S_copy),len(zz))                                                            
fig1 = plt.figure()
#what does add_subplot(111) do? 
ax = fig1.add_subplot(111)
ax.scatter(Mo_coordinate[:,0], Mo_coordinate[:,1], c='r', marker='o')
ax.scatter(S_coordinate[:,0], S_coordinate[:,1], c='b', marker='o')
#ax.scatter(S2_vacancies[:,0], S2_vacancies[:,1], c='b', marker='o')
#ax.scatter(S_vacancies[:,0], S_vacancies[:,1], c='b', marker='o')
#ax.scatter(zz[:,0], zz[:,1], c='b', marker='o')
ax.set_aspect('equal')
plt.title('Figure 1 Mo vs S cooridnates in 2D')


fig2 = plt.figure()
ax2 = fig2.add_subplot(111,projection='3d')
ax2.scatter(Mo_coordinate[:,0], Mo_coordinate[:,1], Mo_coordinate[:,2], c='r', marker='o')
#ax2.scatter(S_coordinate[:,0], S_coordinate[:,1], S_coordinate[:,2], c='b', marker='o')
#ax2.scatter(S2_vacancies[:,0], S2_vacancies[:,1], S2_vacancies[:,2],  c='b', marker='o')
#ax2.scatter(S_vacancies[:,0], S_vacancies[:,1], S_vacancies[:,2],  c='b', marker='o')
ax2.scatter(zz[:,0], zz[:,1], zz[:,2], c='b', marker='o')
#ax2.view_init(elev=90,azim=-60)
plt.title('Figure 2 Mo vs S cooridnates in 3D')


fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
Sulphur_top = zz[zz[:,2]>1]
#Sulphur_top = S_coordinate[S_coordinate[:,2]>1]
#ax3.scatter(Mo_coordinate[:,0], Mo_coordinate[:,1], c='r', marker='o')
#ax.scatter(S_coordinate[:,0], S_coordinate[:,1], c='b', marker='o')
#ax.scatter(S2_vacancies[:,0], S2_vacancies[:,1], c='b', marker='o')
#ax.scatter(S_vacancies[:,0], S_vacancies[:,1], c='b', marker='o')
ax3.scatter(Sulphur_top[:,0], Sulphur_top[:,1], c='b', marker='o')
ax3.set_aspect('equal')
plt.title('Figure 3 plot of top layer of sulphur in MoS2 cell')



fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
clear_zone_top = clear_zone[clear_zone[:,2]>1]
ax4.scatter(clear_zone_top[:,0], clear_zone_top[:,1], c='b', marker='o')
#ax4.scatter(Sulphur_bottom[:,0], Sulphur_bottom[:,1], c='b', marker='o')
ax4.set_aspect('equal')
plt.title('Figure 4 showing holes in top layer of sulphur in MoS2 cell plus extra?')

#need to pick the height
fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
Sulphur_bottom = zz[zz[:,2]<1]
#ax3.scatter(Mo_coordinate[:,0], Mo_coordinate[:,1], c='r', marker='o')
#ax.scatter(S_coordinate[:,0], S_coordinate[:,1], c='b', marker='o')
#ax.scatter(S2_vacancies[:,0], S2_vacancies[:,1], c='b', marker='o')
#ax.scatter(S_vacancies[:,0], S_vacancies[:,1], c='b', marker='o')
ax5.scatter(Sulphur_bottom[:,0], Sulphur_bottom[:,1], c='b', marker='o')
ax5.set_aspect('equal')
plt.title('Figure 5 plot of bottom layer of sulphur in MoS2 cell')



fig6 = plt.figure()
ax6 = fig6.add_subplot(111)
clear_zone_bottom = clear_zone[clear_zone[:,2]<1]
#Sulphur_bottom = zz[zz[:,2]<1]
#ax3.scatter(Mo_coordinate[:,0], Mo_coordinate[:,1], c='r', marker='o')
#ax.scatter(S_coordinate[:,0], S_coordinate[:,1], c='b', marker='o')
#ax.scatter(S2_vacancies[:,0], S2_vacancies[:,1], c='b', marker='o')
ax6.scatter(clear_zone_bottom[:,0], clear_zone_bottom[:,1], c='b', marker='o')
#ax4.scatter(Sulphur_bottom[:,0], Sulphur_bottom[:,1], c='b', marker='o')
ax5.set_aspect('equal')
plt.title('Figure 6 showing the holes on bottom layer plus extra?')

#vacancy_hole
fig7 = plt.figure()
ax7 = fig7.add_subplot(111)
#ax3.scatter(Mo_coordinate[:,0], Mo_coordinate[:,1], c='r', marker='o')
#ax.scatter(S_coordinate[:,0], S_coordinate[:,1], c='b', marker='o')
#ax.scatter(S2_vacancies[:,0], S2_vacancies[:,1], c='b', marker='o')

#vacancy_hole describes coordinates of agglomerated vacancies which have formed a line defect
ax7.scatter(vacancy_hole[:,0], vacancy_hole[:,1], c='b', marker='o')
#VS_holes describes the coordinate of single sulphur vacancies
ax7.scatter(VS_holes[:,0], VS_holes[:,1], c='b', marker='o')

#ax6.scatter(Sulphur_bottom[:,0], Sulphur_bottom[:,1], c='b', marker='o')
ax7.set_aspect('equal')
plt.title('Figure 7 showing the holes (vacancy_hole AND VS_holes)')

plt.show()
'''