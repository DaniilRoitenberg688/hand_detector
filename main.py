import cv2
import mediapipe as mp

from config import *

from detector import Detector

import pandas

from sklearn.neighbors import KNeighborsClassifier

def main(dataset):

    vidcap = cv2.VideoCapture(2)

    detector = Detector(max_hands=2)


    data = pandas.read_csv(all_datasets[dataset])
    results = data['photos_class']
    train = data.drop('photos_class', axis=1)
    knn = KNeighborsClassifier(n_neighbors=8)
    knn.fit(train, results)

    def replace_name(data: dict):
        for key, val in data.items():
            if val in replacement_list:
                data[key] = replacement_list[val]

        return data


    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if not ret:
            break

        new_frame, is_show = detector.find_hand_and_draw(frame, is_draw=True)
        relative = detector.get_relative_coordinates(frame)
        relative = detector.convert_list_to_dataset(relative)
        prediction = detector.predict(relative, model=knn)

        prediction = replace_name(prediction)

        coordinates = detector.get_coordinates(new_frame)
        new_frame = detector.draw_rect(new_frame, coordinates, message=prediction)



        # Display the resized frame
        x, y, z = new_frame.shape
        new_frame = cv2.resize(new_frame, (int(y * 1.5), int(x * 1.5)))

        cv2.namedWindow('Hand', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Hand', int(y * 1.5), int(x * 1.5))
        cv2.imshow('Hand', new_frame)

        # Exit loop by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    vidcap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(1)
