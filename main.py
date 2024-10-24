import cv2
import mediapipe as mp

from config import *

from detector import Detector

import pandas

from sklearn.neighbors import KNeighborsClassifier

vidcap = cv2.VideoCapture(2)

detector = Detector(max_hands=1)

names = header.split(',')

data = pandas.read_csv('dataset.csv')
results = data['photos_class']
train = data.drop('photos_class', axis=1)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train, results)

current = ''

while vidcap.isOpened():
    ret, frame = vidcap.read()
    if not ret:
        break

    new_frame, is_show = detector.find_hand_and_draw(frame, is_draw=True)
    relative = detector.get_relative_coordinates(frame)

    if relative:
        new_data = {}
        for i in range(len(names) - 1):
            new_data[names[i]] = [relative[0][i]]
        new_frame = pandas.DataFrame(new_data)
        a = knn.predict(new_frame)
        if a[0] != current and a[0]:
            current = a[0]
            print(a[0])
    cv2.putText(frame, current,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)

    # Display the resized frame
    cv2.resize(frame, (1000, 1000))
    cv2.imshow('Hand', frame)
    cv2.resizeWindow('Hand', (1000, 1000))

    # Exit loop by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
vidcap.release()
cv2.destroyAllWindows()
