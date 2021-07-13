import cv2
import mediapipe as mp
import numpy as np

max_num_hands = 2
gesture = {
    0:'zero', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'seven', 8:'spiderman', 9:'yeah', 10:'ok',
}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition model
file = np.genfromtxt('gesture_train.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create() # KNN 알고리즘 객체 생성
knn.train(angle, cv2.ml.ROW_SAMPLE, label) # KNN 알고리즘 학습
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    img = cv2.flip(img, 1) #좌우반전 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # mediapipe 는 RGB를 사용하므로

    result = hands.process(img) #전처리와 모델 추론 

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None: # 손감지 True
        rps_result = []

        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))

            angle = np.degrees(angle) # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 4)
            idx = int(results[0][0])

            # Draw gesture result
            if idx in gesture.keys():
                org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))
                cv2.putText(img, text=gesture[idx].upper(), org=(org[0], org[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=1)

                rps_result.append({
                    'rps': gesture[idx],
                    'org': org
                }) # 제스쳐와 각도 저장

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(rps_result) == 2: #이부분 수정
                text = ''
                text1 = ''
                judge = None
                if rps_result[0]['rps'] == 'three':
                    if rps_result[1]['rps'] == 'three' : text= 'Friendly' ; judge = 1 
                elif rps_result[1]['rps'] == 'three':
                    if rps_result[0]['rps'] == 'three' : text= 'Friendly' ; judge = 1 
                else: text1 = 'Enemy'; judge = 0

                if judge == 1:
                    cv2.putText(img, text=text, org=(rps_result[0]['org'][0], rps_result[0]['org'][1] + 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=3)
                else:
                    cv2.putText(img, text=text1, org=(rps_result[1]['org'][0], rps_result[1]['org'][1] + 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255,0, 0), thickness=3)

    cv2.imshow('Camtest', img)
    if cv2.waitKey(1) == ord('q'):
        break
