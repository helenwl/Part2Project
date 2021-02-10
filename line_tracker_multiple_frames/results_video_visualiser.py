import cv2
import os
#import plottingresults_from_all_in_one_helen
#this file makes a video out of a folder of images
results_dir='/media/rob/hdd1/helen/HelenExperimentalCode/Image_generation_MULTEM_largetry/predictions/Brookespredictionseverythingtogetherfolder/linetests/graph_results/'
#results_dir='/media/rob/hdd1/helen/HelenExperimentalCode/Image_generation_MULTEM_largetry/predictions/helenpredictionsalldata/vacancy_mask/cycled_6/'

#uncomment
#batch = 'mg23814-22_cycled_4_minute_00_second_39_frame_0003_to_cycled_4_minute_00_second_41_frame_0007/'
#batch='mg23814-22_cycled_4_minute_00_second_42_frame_0006_to_cycled_4_minute_00_second_47_frame_003/'
batch='mg23814-22_cycled_6_minute_00_second_06_frame_000_to_cycled_6_minute_00_second_13_frame_003/'

image_folder = results_dir + batch
#image_folder=results_dir
#batch_name= '22_cycled_4_minute_0_s_42_f_6_to_minute_0_s_47_f_3'
batch_name= 'all_stiched_together_video'
video_name= image_folder+ f'{batch_name}.avi'

images = [file for file in sorted(os.listdir(image_folder)) if file.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()