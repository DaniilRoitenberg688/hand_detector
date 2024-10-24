import cv2

import mediapipe


class Detector:
    def __init__(self, is_static=False,
                 max_hands=2,
                 model_complexity=1,
                 detection_con=0.7,
                 track_con=0.7):

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
        hands_landmarks = hands.multi_hand_landmarks
        result_coordinates = []
        if hands_landmarks:
            for hand in hands_landmarks:
                hand_coordinates = []
                hand = hand.landmark
                for lm in hand:
                    relative_x = lm.x - hand[0].x
                    relative_y = lm.y - hand[0].y
                    relative_z = lm.z - hand[0].z
                    hand_coordinates.extend([relative_x, relative_y, relative_z])

                result_coordinates.append(hand_coordinates)

        return result_coordinates



if __name__ == '__main__':
    detector = Detector(max_hands=1, is_static=True)
    img = cv2.imread('like/0a1480ec-cb5c-45c3-bead-86defe0e5dd2.jpg')
    print(detector.get_relative_coordinates(img))
    new_img, _ = detector.find_hand_and_draw(img, True)
    cv2.imwrite('test.png', new_img)

