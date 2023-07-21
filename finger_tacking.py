import cv2
import mediapipe as mp
import time


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)


with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index = hand_landmarks.landmark[8]

                cx = int(index.x*1280)
                cy = int(index.y*720)
                if 0<=cx<=1280 and 0<=cy<=720:
                    print(cx,cy)
                    print(results.multi_hand_landmarks.hand_landmarks.landmark[8])


                cv2.putText(
                    image, text='here', org=(cx,cy),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=255, thickness=2)

                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        cv2.imshow('image', image)
        # time.sleep(0)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()