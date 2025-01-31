import torch
import cv2

# 모델 로드 (파인튜닝된 모델이나 COCO 데이터셋 중 선택)
# model_path = '/home/addinedu/venv/objectdetection/cup_detect/yolov5/runs/train/custom_data/weights/best.pt'
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
# model.eval()

# CUDA 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# COCO 데이터셋에서 사전 학습된 YOLOv5m 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  # 'yolov5m', 'yolov5l', 'yolov5x'도 사용 가능
model.eval()  # 모델을 평가 모드로 설정
model.to(device) 

# 타겟 클래스 설정
target_classes = ['cup']

# 신뢰도 임계값 설정
confidence_threshold = 0.1  # 바운더리 안에서만 잡으면 되어 엄청 낮춤

# 테스트 영상
test_path = "/home/addinedu/venv/objectdetection/cup_detect/testing/"
video_file = test_path + "6.mp4"  
cap = cv2.VideoCapture(video_file)

#cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO로 객체 탐지
        results = model(frame)

        # 결과 필터링 (클래스, 신뢰도, 좌표 범위 조건 추가)
        filtered_results = results.pandas().xyxy[0]
        filtered_results = filtered_results[
            (filtered_results['name'].isin(target_classes)) &  # 타겟 클래스
            (filtered_results['confidence'] > confidence_threshold) &  # 신뢰도 조건
            (filtered_results['xmin'] >= 150) &  # x 최소값
            (filtered_results['xmax'] <= 450) &  # x 최대값
            (filtered_results['ymax'] >= 200)  # y 최소값
        ]

        # 감지 결과 표시
        for _, row in filtered_results.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            confidence = row['confidence']
            label = f"{row['name']} {confidence:.2f}"
            
            # 경계 상자와 라벨 표시
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #컵 좌표 
            print(f"Cup detected at: xmin={x1}, ymin={y1}, xmax={x2}, ymax={y2}, confidence={confidence:.2f}")
        # 결과 프레임 표시
        cv2.imshow('YOLOv5 Detection', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(15) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
