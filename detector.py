import cv2
from sklearn.neighbors import KNeighborsClassifier
import mediapipe
from pprint import pprint

from config import *
import pandas

names = header.split(',')


class Detector:
    def __init__(self, is_static=False,
                 max_hands=2,
                 model_complexity=1,
                 detection_con=0.5,
                 track_con=0.5):

        self.mp_hands = mediapipe.solutions.hands
        self.drawing = mediapipe.solutions.drawing_utils

        self.hands_detector = self.mp_hands.Hands(static_image_mode=is_static,
                                                  max_num_hands=max_hands,
                                                  model_complexity=model_complexity,
                                                  min_detection_confidence=detection_con,
                                                  min_tracking_confidence=track_con)

    def find_hand_and_draw(self, img, is_draw):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hands = self.hands_detector.process(img_rgb)

        if hands.multi_hand_landmarks:
            for lm in hands.multi_hand_landmarks:
                if is_draw:
                    self.drawing.draw_landmarks(img, lm, self.mp_hands.HAND_CONNECTIONS)
        return img, hands.multi_hand_landmarks

    def get_relative_coordinates(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hands = self.hands_detector.process(img_rgb)
        hands_landmarks = hands.multi_hand_world_landmarks
        result_coordinates = []
        if hands_landmarks:
            for hand in hands_landmarks:
                hand_coordinates = []
                hand = hand.landmark
                for lm in hand:
                    relative_x = lm.x
                    relative_y = lm.y
                    relative_z = lm.z
                    hand_coordinates.append([relative_x, relative_y, relative_z])

                result_coordinates.append(hand_coordinates)


        hands_labels = hands.multi_handedness
        result = {}
        if hands_labels:
            if len(hands_labels) == 2:
                result = {hands_labels[0].classification[0].label: result_coordinates[0],
                          hands_labels[1].classification[0].label: result_coordinates[1]}
            else:
                result = {hands_labels[0].classification[0].label: result_coordinates[0]}

        return result

    def get_coordinates(self, img):
        bgr_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hands = self.hands_detector.process(bgr_img)
        hands_landmarks = hands.multi_hand_landmarks
        result_coordinates = []
        height, width, deep = img.shape
        if hands_landmarks:
            for hand in hands_landmarks:
                hand_coordinates = []
                hand = hand.landmark
                for lm in hand:
                    hand_coordinates.append([int(lm.x * width), int(lm.y * height), int(lm.z * deep)])

                result_coordinates.append(hand_coordinates)

        hands_labels = hands.multi_handedness
        result = {}
        if hands_labels:
            if len(hands_labels) == 2:
                result = {hands_labels[0].classification[0].label: result_coordinates[0],
                          hands_labels[1].classification[0].label: result_coordinates[1]}
            else:
                result = {hands_labels[0].classification[0].label: result_coordinates[0]}


        return result

    def draw_rect(self, img, coordinates: dict, message: dict):

        for key, hand in coordinates.items():

            by_x = sorted(hand, key=lambda x: x[0])
            by_y = sorted(hand, key=lambda x: x[1])

            start = (by_x[0][0] - 10, by_y[0][1] - 10)
            # print(start[0])
            end = (by_x[-1][0] + 10, by_y[-1][1] + 10)

            img = cv2.rectangle(img, start, end, (255, 0, 0), 3)
            cv2.putText(img, message.get(key, ''),
                        (start[0], start[1] - 5),
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)

        return img

    def convert_list_to_dataset(self, data: dict) -> dict:
        result = {}
        for key, hand in data.items():
            new_data = {}
            for i in range(0, 63, 3):
                new_data[names[i]] = [hand[i // 3][0]]
                new_data[names[i + 1]] = [hand[i // 3][1]]
                # new_data[names[i + 2]] = [hand[i // 3][2]]

            new_data_frame = pandas.DataFrame(new_data)

            result[key] = new_data_frame

        return result

    def predict(self, dataset: dict, model):
        prediction = {}
        for key, data in dataset.items():
            model_prediction = model.predict(data)
            prediction[key] = model_prediction[0]

        return prediction







if __name__ == '__main__':

    data = pandas.read_csv('dataset.csv')
    results = data['photos_class']
    train = data.drop('photos_class', axis=1)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train, results)

    detector = Detector(max_hands=2, is_static=True)
    img = cv2.imread('training_data/test_double_hands/img.png')
    coordinates = detector.get_coordinates(img)
    dataframe = detector.convert_list_to_dataset(coordinates)
    prediction = detector.predict(dataframe, model=knn)

    print(prediction)



    new_img, _ = detector.find_hand_and_draw(img, True)


    new_img = detector.draw_rect(new_img, coordinates, prediction)

    cv2.imwrite('training_data/test_double_hands/test.png', new_img)
