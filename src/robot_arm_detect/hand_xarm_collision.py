import cv2
import torch
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# ✅ YOLOv8 Segmentation 모델 로드
model = YOLO("/home/olivia/collision_ws/src/robot_collision/best.pt")

# ✅ Mediapipe Hand Tracking 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ✅ 웹캠 실행
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ✅ YOLO Segmentation 실행
        results = model(frame, task="segment", conf=0.25)  # 작은 객체도 감지

        # ✅ 로봇팔 Segmentation 테두리만 표시
        robot_masks = []
        for result in results:
            if result.masks is not None:
                for mask in result.masks.xy:
                    mask = np.array(mask, dtype=np.int32)
                    robot_masks.append(mask)
                    cv2.polylines(frame, [mask], isClosed=True, color=(0, 255, 0), thickness=3)  # 테두리만 표시

        # ✅ Mediapipe 손 추적 (hand_tracking.py 방식 유지)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb_frame)

        collision_detected = False  # 충돌 여부

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    h, w, _ = frame.shape
                    hand_x, hand_y = int(landmark.x * w), int(landmark.y * h)  # 손 좌표 변환

                    # ✅ 손 랜드마크 표시 (hand_tracking.py 방식 유지)
                    cv2.circle(frame, (hand_x, hand_y), 5, (0, 0, 255), -1)  # 빨간색 점

                    # ✅ 손 좌표가 로봇팔 테두리 안에 있는지 확인
                    for mask in robot_masks:
                        if cv2.pointPolygonTest(mask, (hand_x, hand_y), False) >= 0:
                            collision_detected = True
                            break

        # ✅ 충돌 감지 메시지 출력 (영어 텍스트 적용)
        if collision_detected:
            cv2.putText(frame, "Collision Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 3, cv2.LINE_AA)  # 빨간색 텍스트 출력

        # ✅ 화면 출력
        cv2.imshow("Robot Arm & Hand Tracking", frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ✅ 웹캠 종료
cap.release()
cv2.destroyAllWindows()
