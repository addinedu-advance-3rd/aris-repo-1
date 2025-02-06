import torch
import cv2
from flask import Flask, render_template, Response

# 모델 로드 (파인튜닝된 모델이나 COCO 데이터셋 중 선택)
# model_path = '/home/addinedu/venv/objectdetection/cup_detect/yolov5/runs/train/custom_data/weights/best.pt'
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
# model.eval()

app = Flask(__name__)

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

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO로 객체 탐지
        results = model(frame)
        filtered_results = results.pandas().xyxy[0]
        filtered_results = filtered_results[
            (filtered_results['name'].isin(target_classes)) &
            (filtered_results['confidence'] > confidence_threshold) &
            (filtered_results['xmin'] >= 150) &
            (filtered_results['xmax'] <= 450) &
            (filtered_results['ymax'] >= 200)
        ]

        # 감지 결과 표시
        for _, row in filtered_results.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = f"{row['name']} {row['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 프레임 JPEG 인코딩 및 전송
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    """메인 페이지 렌더링."""
    return render_template('index.html')

@app.route('/video')
def video():
    """비디오 스트리밍 라우트."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


''' 
라우팅 전 코드 아래 기능들 다 있어야함
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
'''