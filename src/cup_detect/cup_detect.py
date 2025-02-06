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

    def stream_video_test(self):
        self.load_model()
        fps = self.setup_video()
        initial_detected = False
        event_time = None
        break_count = 0
        post_event_frames = int(fps * self.post_event_seconds)

        # self.cap = cv2.VideoCapture(0)

        while True:
            ret, frame = self.cap.read()
            if (frame is None):
                print("stream_video_break-frame is None", flush=True)
                break
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                print(f"Sending frame: {len(frame_bytes)} bytes", flush=True)

                results = self.model(frame)
                df_results = results.pandas().xyxy[0]
                object_outside_boundary = False
                filtered_results = df_results[
                    (df_results['name'].isin(self.target_classes)) &
                    (df_results['confidence'] > self.confidence_threshold) &
                    (df_results['xmin'] >= 150) &
                    (df_results['xmax'] <= 450) &
                    (df_results['ymax'] >= 200)
                ]

        # # 감지 결과 표시
        # for _, row in filtered_results.iterrows():
        #     x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        #     label = f"{row['name']} {row['confidence']:.2f}"
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


                for _, row in df_results.iterrows():
                    if row['name'] in self.target_classes and row['confidence'] > self.confidence_threshold:
                        xmin, xmax = int(row['xmin']), int(row['xmax'])
                        ymin, ymax = int(row['ymin']), int(row['ymax'])
                        confidence = row['confidence']
                        label = f"{row['name']} {confidence:.2f}"

                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        

                print("printing frame", post_event_frames)

                return frame
            else:
                print("Empty frame received")
                break
        self.cap.release()


    def process_frame(self, frame):

            results = self.model(frame)
            df_results = results.pandas().xyxy[0]
            initial_detected = False
            # print("initial_detected", initial_detected, flush=True)
            object_outside_boundary = False

            # Check detections
            for _, row in df_results.iterrows():
                if row['name'] in self.target_classes and row['confidence'] > self.confidence_threshold:
                    xmin, xmax = int(row['xmin']), int(row['xmax'])
                    ymin, ymax = int(row['ymin']), int(row['ymax'])
                    confidence = row['confidence']
                    label = f"{row['name']} {confidence:.2f}"

                    # Draw bounding boxes
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Check if object is initially inside boundary
                    if (xmin >= self.boundary_xmin and xmax <= self.boundary_xmax and
                            ymin >= self.boundary_ymin and ymax <= self.boundary_ymax):
                        if not initial_detected:
                            initial_detected = True
                            print("객체가 처음 바운더리 안에 있습니다.")

                    # Check if object moves outside boundary
                    elif initial_detected and (xmin >= self.outside_xmin and xmax <= self.outside_xmax and
                                            ymin >= self.outside_ymin and ymax <= self.outside_ymax):
                        object_outside_boundary = True
                        if event_time is None:
                            event_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                            print(f"Event detected at: {event_time:.2f} seconds")


                # 
                # Write to output

                    # self.out.write(frame)
                    # self.out.write(frame)

                    # Event handling
                    # if object_outside_boundary:
                    #     print("이벤트 발생: 객체가 바운더리를 벗어났습니다.")
                    #     if break_count == 0:  # Start counting after the event
                    #         break_count += 1

                    # # Draw boundary boxes
                    # cv2.rectangle(frame, (self.boundary_xmin, self.boundary_ymin),
                    #             (self.boundary_xmax, self.boundary_ymax), (255, 0, 0), 2)
                    # cv2.rectangle(frame, (self.outside_xmin, self.outside_ymin),
                    #             (self.outside_xmax, self.outside_ymax), (255, 0, 0), 2)

                    # if event_time and break_count > 0 and break_count < post_event_frames:
                    #     break_count += 1
                    # elif break_count >= post_event_frames:
                    #     break




            return frame


    def detect_and_write(self, fps: float):

        print("detect_and_write", flush=True)
        print("fps : ", fps, flush=True)
        """
        Read frames, perform detection, draw bounding boxes,
        and save frames to the output video.
        Stop recording after the specified post-event frames.
        """

        # Flags and counters
        initial_detected = False
        event_time = None
        break_count = 0
        post_event_frames = int(fps * self.post_event_seconds)

        try:
            print("detect_and_write_try", flush=True)
            

            while True:
                ret, frame = self.cap.read()
                if (frame is None):
                    print("stream_video_break-frame is None", flush=True)
                    break
                if frame is not None:
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    # print(f"Sending frame: {len(frame_bytes)} bytes", flush=True)

                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    print("Empty frame received")
                    break
                if not ret:
                    print("stream_video_break", flush=True)
                    break
                print("stream_video_continue", flush=True)
                # yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')
            self.cap.release()
                        # Run detection

            
            results = self.model(frame)
            df_results = results.pandas().xyxy[0]

            object_outside_boundary = False

            # Check detections
            for _, row in df_results.iterrows():
                if row['name'] in self.target_classes and row['confidence'] > self.confidence_threshold:
                    xmin, xmax = int(row['xmin']), int(row['xmax'])
                    ymin, ymax = int(row['ymin']), int(row['ymax'])
                    confidence = row['confidence']
                    label = f"{row['name']} {confidence:.2f}"

                    # Draw bounding boxes
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Check if object is initially inside boundary
                    if (xmin >= self.boundary_xmin and xmax <= self.boundary_xmax and
                            ymin >= self.boundary_ymin and ymax <= self.boundary_ymax):
                        if not initial_detected:
                            initial_detected = True
                            print("객체가 처음 바운더리 안에 있습니다.", flush=True)

                    # Check if object moves outside boundary
                    elif initial_detected and (xmin >= self.outside_xmin and xmax <= self.outside_xmax and
                                            ymin >= self.outside_ymin and ymax <= self.outside_ymax):
                        object_outside_boundary = True
                        if event_time is None:
                            event_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                            print(f"Event detected at: {event_time:.2f} seconds")


                # 
                # Write to output

                    # self.out.write(frame)
                    # self.out.write(frame)

                    # Event handling
                    if object_outside_boundary:
                        print("이벤트 발생: 객체가 바운더리를 벗어났습니다.")
                        if break_count == 0:  # Start counting after the event
                            break_count += 1

                    # Draw boundary boxes
                    cv2.rectangle(frame, (self.boundary_xmin, self.boundary_ymin),
                                (self.boundary_xmax, self.boundary_ymax), (255, 0, 0), 2)
                    cv2.rectangle(frame, (self.outside_xmin, self.outside_ymin),
                                (self.outside_xmax, self.outside_ymax), (255, 0, 0), 2)

                    if event_time and break_count > 0 and break_count < post_event_frames:
                        break_count += 1
                    elif break_count >= post_event_frames:
                        break

                # else:
                #     print("Empty frame received")
                #     break
                # if not ret:
                #     print("stream_video_break", flush=True)
                #     break
                # print("stream_video_continue", flush=True)

                # Show the detection in a window (optional)
                # cv2.imshow('YOLOv5 Detection', frame)
                # if cv2.waitKey(30) & 0xFF == ord('q'):
                #     break

        finally:
            # Clean up
            self.cap.release()
            self.out.release()
            cv2.destroyAllWindows()

    def run_detection(self):
        print("run_detection", flush=True)

        """
        Main entry point to load the model, set up video I/O,
        run detection, and clean up.
        """
        self.load_model()
        fps = self.setup_video()
        self.detect_and_write(fps)

    def __del__(self):
        """
        Destructor to ensure resources are released if something goes wrong
        and the object is destroyed before run_detection completes.
        """
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()


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
    initial_detected = False
    event_time = None
    break_count = 0
    # post_event_frames = int(fps * post_event_seconds)


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
            print(f"Sending frame: {len(frame_bytes)} bytes", flush=True)

            processed_frame = detector.process_frame(frame)


            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            print("Empty frame received")
            break
        if not ret:
            print("stream_video_break", flush=True)
            break
        print("stream_video_continue", flush=True)
        # yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')
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
