# 드론정찰 및 피아식별
한이음 ICT 4드론 팀입니다.

---
yolo-tiny 를 사용하여 군인을 인식함.
> military-object weigt file github **Click** [here](https://github.com/haris0/military-object) 

* 테스트 
![test_img](https://lab.hanium.or.kr/21_HI025/21_hi025/raw/master/test_img/test_img.jpeg)

* 결과
![result](https://lab.hanium.or.kr/21_HI025/21_hi025/raw/master/test_img/result.jpeg)

> Mediapipe 를 이용한 손가락 인식을 통해 일종의 수신호(바디사인)을 받아 아군/적군을 판단한다. **Click** [here](https://google.github.io/mediapipe/solutions/hands.html)

![img](https://google.github.io/mediapipe/images/mobile/hand_landmarks.png)

이미 학습된 'gesture_train.csv' 파일을 KNN 모델에 넣어 training 시키고
영상을 입력해주고 mediapipe_hand솔루션으로 손가락 마디 좌표를 얻는다.
그리고 학습된 KNN 모델의 출력을 얻고 그것이 무슨 수신호인지 알 수 있다.

![img](https://lab.hanium.or.kr/21_HI025/21_hi025/raw/master/test_img/%EC%86%90%EA%B0%80%EB%9D%BD%EC%9D%B8%EC%8B%9D.png)
