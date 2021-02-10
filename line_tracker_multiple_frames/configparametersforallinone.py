#CHANGE batch to change the set of images we are analysing
'''
batch='mg23814-22_cycled_4_minute_00_second_39_frame_0003_to_cycled_4_minute_00_second_41_frame_0007'
batch='mg23814-22_cycled_4_minute_00_second_42_frame_0006_to_cycled_4_minute_00_second_47_frame_003'
batch='mg23814-22_cycled_6_minute_00_second_06_frame_000_to_cycled_6_minute_00_second_13_frame_003'
batch= 'mg23814-22_cycled_7_minute_0_8_s_f_3_to_cycled_7_minute_0_14_s_f_3'
batch= 'mg23814-22_cycled_5_minute_1_second_29_f_1_to_minute_2_s_4_f_1'
batch = 'mg23814-22_cycled_5_minute_2_second_19_f_0_to_minute_2_s_24_f_1'
batch= 'mg23814-22_cycled_6_minute_0_s_18_f_1_to_minute_0_s_36_f_5'
batch='test_frames'
batch='mg23814-22-5_1_42_0_to_1_43_6'
'''
#n_clusters is the number of line defects we want to detect
#threshold_distance is for counting whether a point lies on a linedefect or not
#threshold_point_distance is for counting a cluster on a line to be an outlier
#threshold_distance_width is for when we are measuring the width of a line defect with a perpendicular line
#initial_line_number: in the first frame, which line do we want to track?

batch ="mg23814-22-6_0_18_1_to_0_28_0"
seegraphs = False
first_run_on_batch = False


if batch =="mg23814-22-5_1_38_6_to_1_49_4":
    reverse= True
    shallow = False
    initial_line_number=0
    n_clusters= 8
    threshold= 30
    minLineLength= 30
    maxLineGap=70
    threshold_distance= 100
    threshold_point_distance= 300
    threshold_distance_width=10
    alike_lines_tolerance=0.15
if batch =="mg23814-22-5_1_29_1_to_1_34_0":
    reverse= True
    shallow = False
    initial_line_number=2
    n_clusters= 7
    threshold= 30
    minLineLength= 30
    maxLineGap=70
    threshold_distance= 100
    threshold_point_distance= 300
    threshold_distance_width=10
    alike_lines_tolerance=0.1
if batch == "mg23814-22-5_1_42_0_to_1_43_6":
    reverse= True
    shallow = False
    initial_line_number=1
    n_clusters= 8
    threshold= 30
    minLineLength= 30
    maxLineGap=90
    threshold_distance= 100
    threshold_point_distance= 350
    threshold_distance_width=10
    alike_lines_tolerance=0.1
if batch =="mg23814-22-5_1_57_2_to_2_3_6":
    #line of interest here is MOVING out of the picture, not growing. SO no need to reverse reading
    reverse= False
    shallow=True
    initial_line_number=3
    n_clusters= 8
    threshold= 30
    minLineLength= 30
    maxLineGap=90
    threshold_distance= 100
    threshold_point_distance= 350
    threshold_distance_width=10
    alike_lines_tolerance=0.1
if batch == "mg23814-22-7_0_8_4_to_0_11_1":
    reverse= False
    shallow=True
    initial_line_number=5
    n_clusters= 10
    threshold= 40
    minLineLength= 15
    maxLineGap=90
    threshold_distance= 100
    threshold_point_distance= 300
    threshold_distance_width=10
    alike_lines_tolerance=0.2
if batch =="mg23814-22-7_0_9_4_to_0_12_7":
    reverse= True
    shallow=True
    initial_line_number=8
    n_clusters= 10
    threshold= 30
    minLineLength= 15
    maxLineGap=90
    threshold_distance= 100
    threshold_point_distance= 300
    threshold_distance_width=10
    alike_lines_tolerance=0.2
if batch =="mg23814-22-7_0_8_7_to_0_14_2":
    reverse= True
    shallow=True
    initial_line_number=5
    n_clusters= 10
    threshold= 40
    minLineLength= 15
    maxLineGap=90
    threshold_distance= 100
    threshold_point_distance= 210
    threshold_distance_width=10
    alike_lines_tolerance=0.2
if batch=="mg23814-22-6_0_7_0_to_0_11_3":
    reverse= True
    shallow=False
    initial_line_number=9
    n_clusters= 10
    threshold= 30
    minLineLength= 15
    maxLineGap=90
    threshold_distance= 150
    threshold_point_distance= 210
    threshold_distance_width=10
    alike_lines_tolerance=0.2
if batch == "mg23814-22-4_0_37_3_to_0_41_7":
    reverse= True
    shallow=True
    initial_line_number=2
    n_clusters= 5
    threshold= 30
    minLineLength= 15
    maxLineGap=90
    threshold_distance= 110
    threshold_point_distance= 400
    threshold_distance_width=10
    alike_lines_tolerance=0.2
if batch == "mg23814-22-4_0_46_2_to_0_47_5":
    reverse= True
    shallow=False
    initial_line_number=2
    n_clusters= 5
    threshold= 30
    minLineLength= 15
    maxLineGap=90
    threshold_distance= 150
    threshold_point_distance= 700
    threshold_distance_width=10
    alike_lines_tolerance=0.1
if batch =="mg23814-22-4_0_41_5_to_0_46_1":
    reverse= True
    shallow=False
    initial_line_number=2
    n_clusters= 5
    threshold= 30
    minLineLength= 15
    maxLineGap=90
    threshold_distance= 90
    threshold_point_distance= 600
    threshold_distance_width=10
    alike_lines_tolerance=0.1
if batch =="mg23814-22-6_0_18_1_to_0_28_0":
    reverse= True
    shallow=False
    initial_line_number=2
    n_clusters= 5
    threshold= 50
    minLineLength= 15
    maxLineGap=80
    threshold_distance= 90
    threshold_point_distance= 600
    threshold_distance_width=10
    alike_lines_tolerance=0.1