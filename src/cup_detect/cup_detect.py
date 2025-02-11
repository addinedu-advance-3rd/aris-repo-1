import cv2
import torch
import time
import os
from flask import Flask, render_template, Response
from flask_cors import CORS
import threading
import queue
import numpy as np
import requests
import subprocess
import sys
import subprocess



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
                #  output_path: str = 'output.avi',
                 target_classes: list = None,
                 confidence_threshold: float = 0.45,
                 boundary_coords: tuple = (180, 420, 180, 350),
                 outside_padding: int = 100,
                 post_event_seconds: int = 5,
                 use_cuda: bool = True):
        print("init_start", flush=True)


        self.video_path = 0
        # self.output_path = output_path
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
        self.out = None
        self.frame_queue = queue.Queue(maxsize=100)


        # CUDA or CPU
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

        # Internal variables
        self.cap = None
        self.model = None
        try:
            # self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='custom_data/cup_detect_coco_finetune/weights/best.pt')
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='/app/shared_folder/best.pt')
            self.model.eval()  # Set model to evaluation mode
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

        print("init_done", flush=True)

    #############################################
    # Define the Saving Thread Function
    #############################################
    # def saving_video_thread(self, frame_queue, output_path, fourcc, fps, frame_size):
    def saving_video_thread(self, frame_queue, output_path, fourcc, fps, frame_size):
        """
        Continuously reads frames from frame_queue and writes them to the output video file.
        A 'None' frame is used as a sentinel to signal the thread to exit.
        """
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        self.out = out
        if not out.isOpened():
            print("Failed to open output video file.")
            return
        frame_count = 0
        while True:
            frame = frame_queue.get()  # blocks until a frame is available
            if frame is None:  # sentinel received: end the thread
                break
            # print("saving_video_thread", flush=True)
            out.write(frame)
            frame_count += 1
        self.out.release()
        print("saving_video_thread_done", flush=True)
        print(f"Frame {frame_count} written to video.", flush=True)
        
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
                    # print(f"xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax}", flush=True)


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
                            # self.out.release()
                            convert_to_mp4()
                            self.frame_queue.put(None)
                            print("saving_video_thread_done_EVENT", flush=True)


                        # ✅ 4. 객체가 바운더리 밖으로 나가면, initial_detected를 False로 재설정하여 다시 감지 가능하도록 함
                        self.initial_detected = False

            return frame

detector = CupBoundaryDetector(
    video_path='0',
    # video_path='testing/6.mp4',
    # output_path='output.avi',
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


    # cap = cv2.VideoCapture('/app/last_10_seconds.avi')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    # fourcc= cv2.cv.CV_FOURCC('m', 'p', '4', 'v')

    # out = cv2.VideoWriter('output.avi', fourcc, 30.0, (600, 480))
    
    save_thread = threading.Thread(target=detector.saving_video_thread, args=(detector.frame_queue, 'output.avi', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4)))))
    save_thread.daemon = True
    save_thread.start()

    if not cap.isOpened():
        print("Failed to open.")
        return  # Or yield nothing


    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps", fps, flush=True)
    if fps == 0:
        fps = 30.0  # fallback if fps is 0
    err_count = 0


    start_time = time.time()

    while True:
        if time.time() - start_time > 15:
            print("Timeout reached: 15 seconds elapsed. Finishing recording.", flush=True)
            break


        ret, frame = cap.read()
        if (frame is None):
            print("stream_video_break-frame is None", flush=True)
            break
        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            # print(f"Sending frame: {len(frame_bytes)} bytes", flush=True)

            processed_frame = detector.process_frame(frame, cap)
             # ✅ BGR 포맷이 아닐 경우 변환
            if len(processed_frame.shape) == 2:  # Grayscale인 경우
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
            elif processed_frame.shape[2] == 4:  # RGBA인 경우
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGBA2BGR)

            try :
                if processed_frame is not None and processed_frame.shape[0] > 0 and processed_frame.shape[1] > 0:
                    try:
                        detector.frame_queue.put(processed_frame, block=False)
                        # print(f"📌 Frame added to queue. Queue size: {detector.frame_queue.qsize()}", flush=True)
                    except Exception as e:
                        print(f"Error: {e}")
                        err_count += 1
                        if err_count > 10:
                            print("Error: 10 frames in a row", flush=True)
                            convert_to_mp4()
                            # video_recording_done()
                            detector.frame_queue.put(None)
                            break
                    

                else:
                    print("⚠️ Invalid processed_frame detected! Skipping...", flush=True)
                # dummy_frame = np.zeros((600, 480, 3), dtype=np.uint8)
                detector.frame_queue.put(processed_frame, block=False)
            except Exception as e:
                print(f"Error: {e}")

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            print("Empty frame received")
            break
        if not ret:
            print("stream_video_break", flush=True)
            break
        ret2, buffer = cv2.imencode('.jpg', processed_frame)

        if not ret2:
            print("Failed to encode frame.")
            break

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n')
    cap.release()
    detector.frame_queue.put(None)  # Signal the saving thread to exit
    save_thread.join()
    time.sleep(1)
    print("VIDEO RECORDING AND SAVING DONE", flush=True)
    convert_to_mp4()
    # video_recording_done()
# request post video recording done 

def convert_to_mp4(input_file="output.avi", output_file="/app/video_src/output.mp4"):
    """Converts AVI file to MP4 using FFmpeg"""
    try:
        output_dir =  os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"✅ Created output directory: {output_dir}")


        print("convert_to_mp4", flush=True)
        subprocess.run([
            "ffmpeg","-y", "-i", input_file, "-vcodec", "libx264", "-acodec", "aac", output_file
        ], check=True)
        print(f"✅ Conversion to MP4 successful: {output_file}")
        video_recording_done()
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg conversion failed: {e}")



def video_recording_done():
    # convert_to_mp4()
    """
    Sends a 'done' status to the Node.js GUI container by making a POST request.
    The endpoint is assumed to be 'http://gui_service:3001/video_recording_done'.
    """
    print("video_recording_done_post", flush=True)
    url = 'http://gui_service:3001/video_recording_done'
    # url = 'http://127.0.0.1:3001/video_recording_done'
    payload = {"status": "done"}
    headers = {'Content-Type': 'application/json'}
    try:
        print("video_recording_done_post_try", flush=True)
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()  # Raise an error if the status is not 200-299
        print("Successfully sent done status to Node.js GUI container.", flush=True)
    except requests.exceptions.RequestException as e:
        print(f"Error sending done status: {e}", flush=True)


if __name__ == "__main__":
    

    app.run(host='0.0.0.0', port=6000, debug=True, threaded=True)
