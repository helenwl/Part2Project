from all_in_one_helen import *
import cv2
import os
import pandas as pd 


print('line defect width list:', line_defect_width_list)
number_of_frames= len(os.listdir(path))

x = list(range(0,number_of_frames))
x = [i * 0.125 for i in x]

#SAVE RESULTS AS CSV WITH ALL NECESSARY PARAMETERS
#batch_name= '22_cycled_4_minute_0_s_42_f_6_to_minute_0_s_47_f_3'
batch_name= '22_cycled_6_minute_0_s_06_f_0_to_minute_0_s_13_f_3'

results_data = pd.DataFrame()
results_data.insert(0, 'frame', sorted(os.listdir(path)))
results_data.insert(1, 'time/s', x)
results_data.insert(2, 'line length/pixels', line_length_list)
results_data.insert(3, 'number of vacancies', number_of_vacancies_list)
results_data.insert(4, 'initial lone number', initial_line_number)
results_data.insert(5, 'n_clusters', n_clusters)
results_data.insert(6, 'hough threshold', threshold)
results_data.insert(7, 'hough minlinelength', minLineLength)
results_data.insert(8, 'hough maxLineGap', maxLineGap)
results_data.insert(9, 'threshold distance', threshold_distance)
results_data.insert(10, 'threshold point distance', threshold_point_distance)
results_data.insert(11, 'threshold width distance', threshold_distance_width)


results_data.to_csv(results_dir+ f'{batch_name}_{initial_line_number}.csv')

#REPEAT BUT FOR LINE LENGTH
#calculating line of best fit, assumiing linearity
m, c = np.polyfit(x, line_length_list, 1)
y_best_fit = []
for i in range(len(x)):
    y_best_fit.append(m*x[i]+c)
#print(y_best_fit)
correlation_matrix = np.corrcoef(x, line_length_list)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
plt.plot(x, y_best_fit, label=f'Line of best fit: y={m}x+{c}. \n R value: {r_squared}')
plt.plot(x, line_length_list, label='Experimental Data')
plt.xlabel('Time/seconds')
plt.ylabel('Line Length/pixels')
plt.title(f'Line length vs Time \n tracking line number: {initial_line_number}. \n 39s to 41s')
plt.legend(loc='upper left')
plt.savefig(results_dir + 'length_of_line_defect.png')
plt.show()

#PLOTTING FOR VACANCY NUMBER
#calculating line of best fit, assumiing linearity
m, c = np.polyfit(x, number_of_vacancies_list, 1)
y_best_fit = []
for i in range(len(x)):
    y_best_fit.append(m*x[i]+c)
#print(y_best_fit)

correlation_matrix = np.corrcoef(x, number_of_vacancies_list)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2

plt.plot(x, y_best_fit, label=f'Line of best fit: y={m}x+{c}. \n R value: {r_squared}')
plt.plot(x, number_of_vacancies_list, label='Experimental Data')
plt.xlabel('Time/seconds')
plt.ylabel('Number of vacancies in line')
plt.title(f'No. of Vacancies vs Time \n tracking line number: {initial_line_number}. \n 39s to 41s')
#plt.xticks(np.arange(0, number_of_frames+1, step=1))
plt.legend(loc='upper left')
plt.savefig(results_dir + 'number_of_vacancies.png')
plt.show()

#PLOTTING FOR MAX LINE DEFECT WIDTH
#calculating line of best fit, assumiing linearity
m, c = np.polyfit(x, line_defect_width_list, 1)
y_best_fit = []
for i in range(len(x)):
    y_best_fit.append(m*x[i]+c)
#print(y_best_fit)
correlation_matrix = np.corrcoef(x, line_defect_width_list)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
plt.plot(x, y_best_fit, label=f'Line of best fit: y={m}x+{c}. \n R value: {r_squared}')
plt.plot(x, line_defect_width_list, label='Experimental Data')
plt.xlabel('Time/seconds')
plt.ylabel('Max Line Width')
plt.title(f'Max Line Width vs Time \n tracking line number: {initial_line_number}. \n 39s to 41s')
plt.legend(loc='upper left')
plt.savefig(results_dir + 'max_width_of_line_defect.png')
plt.show()

#PLOTTING FOR AVERAGE LINE DEFECT WIDTH
#calculating line of best fit, assumiing linearity
m, c = np.polyfit(x, line_defect_average_width_list, 1)
y_best_fit = []
for i in range(len(x)):
    y_best_fit.append(m*x[i]+c)
#print(y_best_fit)
correlation_matrix = np.corrcoef(x, line_defect_average_width_list)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
plt.plot(x, y_best_fit, label=f'Line of best fit: y={m}x+{c}. \n R value: {r_squared}')
plt.plot(x, line_defect_average_width_list, label='Experimental Data')
plt.xlabel('Time/seconds')
plt.ylabel('Average Line Width')
plt.title(f'Average Line Width vs Time \n tracking line number: {initial_line_number}. \n 39s to 41s')
plt.legend(loc='upper left')
plt.savefig(results_dir + 'average_width_of_line_defect.png')
plt.show()
