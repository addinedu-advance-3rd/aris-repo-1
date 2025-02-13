import cv2
import torch
import time
import os
from flask import Flask, render_template, Response, jsonify
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

video_status = {"status": "not done"}
# app.config["video_status"] = {"status": "not done"}
cap = None


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

        # global cap
        # Internal variables
        self.cap = None
        self.model = None
        try:
            # self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='custom_data/cup_detect_coco_finetune/weights/best.pt')
            torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
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

        # ✅ 프레임이 None이거나 크기가 다르면 예외 처리
            if frame is None or frame.shape[0] != frame_size[1] or frame.shape[1] != frame_size[0]:
                print(f"⚠️ Frame size mismatch! Expected: {frame_size}, Got: {frame.shape if frame is not None else 'None'}")
                continue

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

                            # 컵 가져감 이벤트 발생

                            # 이벤트 발생 시 비디오 저장 
                            convert_to_mp4(status="TAKEN")
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
    while not detector.frame_queue.empty():
        # 큐가 비어있지 않으면 큐에서 프레임 가져오기
        detector.frame_queue.get()


    
    # global cap
    cap = cv2.VideoCapture(1)
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
    if hasattr(detector, "save_thread") and detector.save_thread.is_alive():
        print("⚠️ Stopping previous save_thread...", flush=True)
        detector.frame_queue.put(None)  # ✅ 현재 실행 중인 쓰레드 종료 신호
        detector.save_thread.join()  # ✅ 기존 쓰레드가 종료될 때까지 대기

    # ✅ 새로운 저장 스레드 시작
    detector.save_thread = threading.Thread(
        target=detector.saving_video_thread,
        args=(detector.frame_queue, 'output.avi', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    )
    detector.save_thread.daemon = True
    detector.save_thread.start()




    # save_thread = threading.Thread(target=detector.saving_video_thread, args=(detector.frame_queue
    # , 'output.avi', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4)))))
    # save_thread.daemon = True
    # save_thread.start()



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
        # if time.time() - start_time > 20:
        if False :
            print("XXXXXXXXX.", flush=True)
        #     print("Timeout reached: 20 seconds elapsed. Finishing recording.", flush=True)
        #     convert_to_mp4(status="NOT TAKEN")
        #     break

        else : 
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
                                convert_to_mp4(status="NOT TAKEN")
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
    detector.save_thread.join()
    time.sleep(1)
    print("VIDEO RECORDING AND SAVING DONE", flush=True)
    # convert_to_mp4(status = "NOT TAKEN")
    # video_recording_done()
# request post video recording done 

def notify_control_service(status):
    print("notify_control_service", flush=True)
    print (status, flush=True)
    url = 'http://control_service:8080/ice_cream_taken'
    payload = {"status": status}
    headers = {'Content-Type': 'application/json'}
    try:
        print("notify_control_service_try", flush=True)
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status() 
        print ("notify_control_service_response", response.json(), flush=True) # Raise an error if the status is not 200-299
    
    except Exception as e:
        print(f"Error sending done status: {e}", flush=True)


def convert_to_mp4(input_file="output.avi", output_file="/app/video_src/output.mp4", status="NOT TAKEN"):
    """Converts AVI file to MP4 using FFmpeg"""
    try:
        # 컵 가져감 이벤트 발생 시 컨트롤 서비스에 알림
        print(f"convert_to_mp4 , status: {status}", flush=True)
        notify_control_service(status)
        output_dir =  os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"✅ Created output directory: {output_dir}")


        avi_cut = "/app/video_src/cut_output.avi"
        # 마지막 10초 영상 추출
        extract_last_10_seconds(input_file, avi_cut, output_fps=30)

        final_output_file = "/app/video_src/output.mp4"

        print("convert_to_mp4", flush=True)
        subprocess.run([
            "ffmpeg","-y", "-i", avi_cut, "-vcodec", "libx264", "-acodec", "aac", "-r", "30", final_output_file
        ], check=True)
        print(f"✅ Conversion to MP4 successful: {final_output_file}")
        print(f"✅ 10초 영상 MP4 변환 성공: {final_output_file}")
        video_recording_done()
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg conversion failed: {e}")



def video_recording_done():
    """
    Sends a 'done' status to the Node.js GUI container by making a POST request.
    """
    print("video_recording_done_post", flush=True)
    
    url = 'http://gui_service:3001/video_recording_done'
    payload = {"status": "done"}  # ✅ 직접 서버로 상태 전송
    
    headers = {'Content-Type': 'application/json', 'Cache-Control': 'no-cache'}
    
    try:
        print("video_recording_done_post_try", flush=True)
        response = requests.post(url, json=payload, timeout=5, headers=headers)
        response.raise_for_status()
        print("✅ Successfully sent 'done' status to Node.js GUI container.", flush=True)
    except requests.exceptions.RequestException as e:
        print(f"❌ Error sending 'done' status: {e}", flush=True)




@app.route("/stop_camera", methods=['POST'])
def stop_camera():
    # global cap
    if detector.cap is None:
        print("[ERROR] ❌ cap is None. Camera might not have been started.", flush=True)
        return jsonify({"error": "Camera was never started"}), 400

    if not detector.cap.isOpened():
        print("[ERROR] ❌ cap is not opened. It might be already released.", flush=True)
        return jsonify({"error": "Camera is not opened"}), 400

    detector.cap.release()
    detector.cap = None
    print("[INFO] ✅ Camera successfully stopped.", flush=True)
    return jsonify({"message": "Camera stopped"}), 200


def extract_last_10_seconds(input_file, output_file, output_fps=30):
    cap = cv2.VideoCapture(input_file)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 마지막 10초 시작 프레임 계산
    last_10_seconds_start_frame = max(0, total_frames - int(original_fps * 10))

    # 새로운 출력 파일 생성
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MJPG'), output_fps, (frame_width, frame_height))

    # 비디오 포인터를 마지막 10초 시작 프레임으로 이동
    cap.set(cv2.CAP_PROP_POS_FRAMES, last_10_seconds_start_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ 마지막 10초 영상 추출 완료: {output_file}")


if __name__ == "__main__":
    

    app.run(host='0.0.0.0', port=6000, debug=True, threaded=True)
