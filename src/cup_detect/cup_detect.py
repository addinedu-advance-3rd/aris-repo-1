import cv2
import torch
import time
import os
from flask import Flask, render_template, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


class CupBoundaryDetector:
    """
    Class to handle cup detection within a specified boundary and
    create an output video that stops recording after a certain event is detected.
    """

    def __init__(self,
                #  video_path: str,
                 video_path: int,
                 output_path: str = 'output.avi',
                 target_classes: list = None,
                 confidence_threshold: float = 0.45,
                 boundary_coords: tuple = (180, 420, 180, 350),
                 outside_padding: int = 100,
                 post_event_seconds: int = 5,
                 use_cuda: bool = True):
        print("init_start", flush=True)


        self.video_path = 0
        self.output_path = output_path
        self.target_classes = target_classes if target_classes else ['cup']
        self.confidence_threshold = confidence_threshold

        # Boundary coordinates
        self.boundary_xmin, self.boundary_xmax, self.boundary_ymin, self.boundary_ymax = boundary_coords
        self.outside_xmin = self.boundary_xmin - outside_padding
        self.outside_xmax = self.boundary_xmax + outside_padding
        self.outside_ymin = self.boundary_ymin - outside_padding
        self.outside_ymax = self.boundary_ymax + outside_padding

        # Event detection settings
        self.post_event_seconds = post_event_seconds
        self.event_time = None
        self.initial_detected = False
        self.object_outside_boundary = False


        # CUDA or CPU
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

        # Internal variables
        self.cap = None
        self.out = None
        self.model = None
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='custom_data/cup_detect_coco_finetune/weights/best.pt')
            self.model.eval()  # Set model to evaluation mode
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

        print("init_done", flush=True)



    def process_frame(self, frame, cap):

            results = self.model(frame)
            df_results = results.pandas().xyxy[0]
            self.cap = cap
            # self.initial_detected = False
            # self.object_outside_boundary = False
            # print("initial_detected", initial_detected, flush=True)
            # event_time = None
            # ✅ 사용자가 설정한 바운더리 박스 (파란색)
            overlay = frame.copy()
            alpha = 0.3  # 투명도 설정 (0: 완전 투명, 1: 완전 불투명)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # ✅ 블렌딩 적용


            # ✅ 바운더리 박스 그리기
            # 테스트 후 주석 처리 필요
            cv2.rectangle(frame, (self.boundary_xmin, self.boundary_ymin),
                        (self.boundary_xmax, self.boundary_ymax), (255, 0, 0), 2)
            cv2.rectangle(frame, (self.outside_xmin, self.outside_ymin),
                        (self.outside_xmax, self.outside_ymax), (0, 255, 0), 2)


            cv2.putText(frame, "Detection Zone", (self.boundary_xmin, self.boundary_ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # ✅ 객체 감지 확인
            # if df_results.empty:
            #     print("❌ No objects detected!", flush=True)
            #     return frame

            # print(df_results[['name', 'confidence']], flush=True)

            for _, row in df_results.iterrows():
                if df_results.empty:
                    return frame   # 데이터프레임이 비어 있으면 다음 루프로 넘어감
                if row['name'] in self.target_classes and row['confidence'] > self.confidence_threshold:
                    xmin, xmax = int(row['xmin']), int(row['xmax'])
                    x_center = (xmin + xmax) / 2
                    ymin, ymax = int(row['ymin']), int(row['ymax'])
                    y_center = (ymin + ymax) / 2
                    confidence = row['confidence']
                    label = f"{row['name']} {confidence:.2f}"

                    #  if not empty dataframe
                    # if len(xmin) > 0:
                    print(f"xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax}", flush=True)


                    # ✅ 1. 객체가 처음 "바운더리 안"에 들어오면 감지 시작
                    if (x_center >= self.boundary_xmin and x_center <= self.boundary_xmax and
                            y_center >= self.boundary_ymin and y_center <= self.boundary_ymax):
                        if not self.initial_detected:  # ✅ 최초 진입 감지
                            self.initial_detected = True
                            print("📌 객체가 처음 바운더리 안에 들어왔습니다.")

                        # ✅ 3. 객체가 다시 안으로 들어오면, object_outside_boundary를 False로 설정
                        self.object_outside_boundary = False

                    # ✅ 2. 객체가 바깥 바운더리로 나가면 이벤트 감지
                    elif self.initial_detected and (
                            x_center < self.outside_xmin or x_center > self.outside_xmax or
                            y_center < self.outside_ymin or y_center > self.outside_ymax):
                        if not self.object_outside_boundary:  # ✅ 바깥 바운더리로 처음 나갔을 때만 이벤트 발생
                            self.object_outside_boundary = True
                            self.event_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                            print(f"🚨 이벤트 발생: 객체가 바운더리를 벗어났습니다. (Event Time: {self.event_time:.2f} 초)")

                        # ✅ 4. 객체가 바운더리 밖으로 나가면, initial_detected를 False로 재설정하여 다시 감지 가능하도록 함
                        self.initial_detected = False

            return frame




def extract_last_10_seconds(input_file: str, output_file: str, output_fps: float = 30.0) -> None:

#     Extract the last 10 seconds from a video and write it to output_file.
    if not os.path.exists(input_file):
        print(f"Error: The file {input_file} does not exist.")
        return

    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"Error: Could not open {input_file}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        print("Warning: Original FPS is invalid, defaulting to 30.0")
        original_fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_width == 0 or frame_height == 0:
        print("Error: Invalid frame size. Cannot extract last 10 seconds.")
        cap.release()
        return

    # Calculate start frame for the last 10 seconds
    last_10_seconds_start_frame = max(0, total_frames - int(original_fps * 10))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, output_fps, (frame_width, frame_height))

    # Position the video capture at the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, last_10_seconds_start_frame)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
    finally:
        cap.release()
        out.release()

detector = CupBoundaryDetector(
    video_path='0',
    # video_path='testing/6.mp4',
    output_path='output.avi',
    target_classes=['cup'],
    confidence_threshold=0.45,
    boundary_coords=(180, 420, 180, 350),  # (xmin, xmax, ymin, ymax)
    outside_padding=100,
    post_event_seconds=5,
    use_cuda=True
)
# detector.run_detection()



@app.route('/video')
def video():
    print("video", flush=True)

    return Response(stream_video(), mimetype='multipart/x-mixed-replace; boundary=frame')


def stream_video():
    cap = cv2.VideoCapture(0)
    # initial_detected = False
    # event_time = None
    break_count = 0
    # post_event_frames = int(fps * post_event_seconds)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (600, 480))
    


    # cap = cv2.VideoCapture('/app/last_10_seconds.avi')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Failed to open.")
        return  # Or yield nothing


    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps", fps, flush=True)
    if fps == 0:
        fps = 30.0  # fallback if fps is 0


    while True:
        ret, frame = cap.read()
        if (frame is None):
            print("stream_video_break-frame is None", flush=True)
            break
        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            # print(f"Sending frame: {len(frame_bytes)} bytes", flush=True)

            processed_frame = detector.process_frame(frame, cap)
            if processed_frame is not None:
                out.write(processed_frame)
            elif processed_frame is None:
                print("processed_frame is None", flush=True)
             # ✅ BGR 포맷이 아닐 경우 변환
            if len(processed_frame.shape) == 2:  # Grayscale인 경우
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
            elif processed_frame.shape[2] == 4:  # RGBA인 경우
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGBA2BGR)



            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            print("Empty frame received")
            break
        if not ret:
            print("stream_video_break", flush=True)
            break
        # print("stream_video_continue", flush=True)
        # yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')
        out.write(processed_frame)
        ret2, buffer = cv2.imencode('.jpg', processed_frame)

        if not ret2:
            print("Failed to encode frame.")
            break

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n')
    cap.release()




if __name__ == "__main__":
    
    # 테스트 코드
    # detector.run_detection()
    
    app.run(host='0.0.0.0', port=6000, debug=True)
    # Example usage

    # Run the detection

    # Extract the last 10 seconds
