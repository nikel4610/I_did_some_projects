# 드론정찰 및 피아식별
한이음 ICT 4드론 팀입니다.

---
yolo-tiny 를 사용하여 군인을 인식함.
> military-object weigt file github **Click** [here](https://github.com/haris0/military-object) 

* 테스트   
* ![test_img](https://user-images.githubusercontent.com/73810942/171082872-88e87c4d-2e16-419e-a63c-050c557bbdb2.jpeg)



* 결과  
* ![result](https://user-images.githubusercontent.com/73810942/171082899-7fc7de83-67fc-41a4-a6fc-9cf18f312477.jpeg)



> Mediapipe 를 이용한 손가락 인식을 통해 일종의 수신호(바디사인)을 받아 아군/적군을 판단한다. **Click** [here](https://google.github.io/mediapipe/solutions/hands.html)

![img](https://google.github.io/mediapipe/images/mobile/hand_landmarks.png)

이미 학습된 'gesture_train.csv' 파일을 KNN 모델에 넣어 training 시키고
영상을 입력해주고 mediapipe_hand솔루션으로 손가락 마디 좌표를 얻는다.
그리고 학습된 KNN 모델의 출력을 얻고 그것이 무슨 수신호인지 알 수 있다.

* 예시
* ![손가락인식](https://user-images.githubusercontent.com/73810942/171083004-f01499a6-616c-4dd5-a901-c2e7e44fefcb.png)
