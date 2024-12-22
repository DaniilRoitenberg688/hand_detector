import csv
import os
import cv2
from urllib3.filepost import writer

from detector import Detector
from config import *



def create_dataset(name, directories, base_directory, files_col, mode):
    detector = Detector(is_static=True, max_hands=1)
    with open(name, mode) as file:
        writer = csv.writer(file)

        if mode == 'w':
            writer.writerow(list(filter(lambda x: 'z' not in x, header.split(','))))
        if directories:
            for directory in directories:
                counter = 0
                for filename in os.listdir(f'{base_directory}/{directory}'):
                    if files_col <= counter:
                        break
                    img = cv2.imread(f'{base_directory}/{directory}/{filename}')
                    relative = detector.get_relative_coordinates(img)
                    if relative:
                        line = []

                        for key, val in relative.items():
                            for i in val:
                                line.append(i[0])
                                line.append(i[1])
                                # line.append(i[2])
                        line.append(directory)
                        writer.writerow(line)


                    counter += 1



if __name__ == '__main__':
    a = input('sure?')
    if a == 'yes':
        create_dataset(name='gestures_dataset.csv', directories=gestures,
                       base_directory='./training_data/gestures', files_col=333, mode='w')
        create_dataset(name='fingers_dataset.csv', directories=fingers,
                       base_directory='./training_data/fingers', files_col=333, mode='w')