import csv
import os
import cv2
from urllib3.filepost import writer

from detector import Detector
from config import header



def create_dataset(directories, files_col):
    detector = Detector(is_static=True, max_hands=1)
    with open('dataset.csv', 'w') as file:
        writer = csv.writer(file)

        writer.writerow(header.split(','))

        if directories:
            for directory in directories:
                counter = 0
                for filename in os.listdir(directory):
                    if files_col <= counter:
                        break
                    img = cv2.imread(f'{directory}/{filename}')
                    relative = detector.get_relative_coordinates(img)
                    if relative:
                        relative[0].append(directory)
                        writer.writerow(relative[0])
                    counter += 1


def add_into_dataset(directories, files_col):
    detector = Detector(is_static=True, max_hands=1)
    with open('dataset.csv', 'a') as file:
        writer = csv.writer(file)
        if directories:
            for directory in directories:
                counter = 0
                for filename in os.listdir(directory):
                    if files_col <= counter:
                        break
                    img = cv2.imread(f'{directory}/{filename}')
                    relative = detector.get_relative_coordinates(img)
                    if relative:
                        relative[0].append(directory)
                        writer.writerow(relative[0])
                    counter += 1


if __name__ == '__main__':
    create_dataset(['dislike', 'like', 'rock', 'call', 'ok', 'peace'], files_col=100)
