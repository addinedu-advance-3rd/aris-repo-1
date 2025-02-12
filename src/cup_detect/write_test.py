import cv2
import torch
import time

# CUDA 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
model.eval()
model.to(device)

# 설정
target_classes = ['cup']
confidence_threshold = 0.3
# test_path = "/home/addinedu/venv/objectdetection/cup_detect/testing/"
# video_file = test_path + "2.mp4"
# cap = cv2.VideoCapture(video_file)
cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30.0


frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# 녹화 시작
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# 영역 설정
boundary_xmin = 80
boundary_xmax = 460
boundary_ymin = 150
boundary_ymax = 400
outside_xmin = boundary_xmin - 100
outside_xmax = boundary_xmax + 100
outside_ymin = boundary_ymin - 100
outside_ymax = boundary_ymax + 100

# 이벤트 발생 시간 기록
event_time = None
post_event_frames = int(fps * 5)  # 이벤트 후 5초간의 프레임 수
break_count = 0  # Break count initialized here

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        results = results.pandas().xyxy[0]

        object_outside_boundary = False
        for _, row in results.iterrows():
            if row['name'] in target_classes and row['confidence'] > confidence_threshold:
                xmin, xmax, ymin, ymax = int(row['xmin']), int(row['xmax']), int(row['ymin']), int(row['ymax'])
                if (xmin >= outside_xmin and xmax <= outside_xmax and ymin >= outside_ymin and ymax <= outside_ymax):
                    if not (xmin >= boundary_xmin and xmax <= boundary_xmax and ymin >= boundary_ymin and ymax <= boundary_ymax):
                        object_outside_boundary = True
                        if event_time is None:
                            event_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # 이벤트 시간 기록
                            print(f"Event detected at: {event_time:.2f} seconds")
                    else:
                        print("객체가 바운더리 안에 있습니다.")
        out.write(frame)

        if object_outside_boundary:
            print("이벤트 발생: 객체가 바운더리를 벗어났습니다.")
            if break_count == 0:  # Start counting after the event
                break_count += 1

        #바운더리 박스 시각화
        cv2.rectangle(frame, (boundary_xmin, boundary_ymin), (boundary_xmax, boundary_ymax), (255, 0, 0), 2)
        #경계 범위 박스 
        cv2.rectangle(frame, (outside_xmin,outside_ymin), (outside_xmax, outside_ymax), (255,0,0), 2)
        
        if event_time and break_count > 0 and break_count < post_event_frames:
            break_count += 1
        elif break_count >= post_event_frames:
            break

        cv2.imshow('YOLOv5 Detection', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def extract_last_10_seconds(input_file, output_file, output_fps=30):
    # 영상 읽기
    cap = cv2.VideoCapture(input_file)

    # 영상 정보 가져오기
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 마지막 10초의 시작 프레임 계산 (원본 FPS 기준)
    last_10_seconds_start_frame = max(0, total_frames - int(original_fps * 10))

    # 비디오 쓰기 설정 (30 FPS로 저장)
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'XVID'), output_fps, (frame_width, frame_height))

    # 시작 프레임으로 이동
    cap.set(cv2.CAP_PROP_POS_FRAMES, last_10_seconds_start_frame)

    # 마지막 10초 프레임 추출 및 저장
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    # 리소스 해제
    cap.release()
    out.release()

# 함수 호출 예시
extract_last_10_seconds('output.avi', 'last_10_seconds.avi', output_fps=30)
print("마지막 10초 영상 추출 완료")