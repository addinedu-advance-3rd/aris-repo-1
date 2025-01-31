import torch
import cv2

# 파인튜닝된 모델 로드, 'best.pt' 파일 경로 확인 필요 -> exp 번호 잘 확인
# model_path = '/home/addinedu/venv/objectdetection/cup_detect/yolov5/runs/train/custom_data/weights/best.pt'
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
# model.eval()  # 모델을 평가 모드로 설정

# COCO 데이터셋에서 사전 학습된 YOLOv5m 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  # 'yolov5m', 'yolov5l', 'yolov5x'도 사용 가능
model.eval()  # 모델을 평가 모드로 설정

# 타겟 클래스 설정
target_classes = ['cup']

# 신뢰도 임계값 설정
confidence_threshold = 0.3  # 높게 설정하여 잘못된 감지를 제거

# 테스트 영상
test_path = "/home/addinedu/venv/objectdetection/cup_detect/testing/"
video_file = test_path + "7.mp4"  
cap = cv2.VideoCapture(video_file)

# 웹캠 설정
#cap = cv2.VideoCapture(0)  # 웹캠 사용

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO로 객체 탐지
        results = model(frame)

        # 결과 필터링
        filtered_results = results.pandas().xyxy[0]
        filtered_results = filtered_results[(filtered_results['name'].isin(target_classes)) &
                                            (filtered_results['confidence'] > confidence_threshold)]

        # 감지 결과 표시
        for _, row in filtered_results.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            confidence = row['confidence']
            label = f"{row['name']} {confidence:.2f}"

            # 경계 상자와 라벨 표시
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 결과 프레임 표시
        cv2.imshow('YOLOv5 Detection', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()