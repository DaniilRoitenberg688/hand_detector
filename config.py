import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (0, 150)
fontScale = 2
fontColor = (0, 0, 255)
thickness = 2
lineType = 1

replacement_list = {'three2': 'three',
                    'two2': 'two',
                    'stop': 'wait'}

fingers = ['five', 'four', 'one', 'three', 'three2', 'two', 'two2', 'zero']
gestures = ['call', 'dislike', 'like', 'ok', 'peace', 'rock', 'stop']

all_datasets = ['fingers_dataset.csv', 'gestures_dataset.csv']

header = '0_x,0_y,0_z,1_x,1_y,1_z,2_x,2_y,2_z,3_x,3_y,3_z,4_x,4_y,4_z,5_x,5_y,5_z,6_x,6_y,6_z,7_x,7_y,7_z,8_x,8_y,8_z,9_x,9_y,9_z,10_x,10_y,10_z,11_x,11_y,11_z,12_x,12_y,12_z,13_x,13_y,13_z,14_x,14_y,14_z,15_x,15_y,15_z,16_x,16_y,16_z,17_x,17_y,17_z,18_x,18_y,18_z,19_x,19_y,19_z,20_x,20_y,20_z,photos_class'
