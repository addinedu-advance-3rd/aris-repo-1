import os
import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO
# from yolov5 import YOLO  # YOLO 모델 로드
import mediapipe as mp
import threading
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import requests


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # This will automatically add the header "Access-Control-Allow-Origin: *" to every response


sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI

class A_Circle_Arm():

    # 싱글톤 패턴 적용
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(A_Circle_Arm, cls).__new__(cls)

        return cls._instance

    def __init__(self, arm_ip, app):
        if not hasattr(self, "initialized"):
            self.end_check_point = False
            self.arm_ip = arm_ip
            self.app = app
            self.ice_cream_taken = False
            self.arm = None
            self.connect_to_arm()
            self.ice_cream_event = threading.Event()
            self.running = True
            self.last_no_collision_time = None  # 최근 충돌이 없었던 시간을 기록
            self.stop_collision_thread_status = False
            self.stop_collision_handler_thread_status = False
            self.app.add_url_rule(
                '/select_toppings',
                view_func=self.select_toppings,
                methods=['POST', 'OPTIONS']
            )

            self.collision_detected = False
            self.model = YOLO("best_robot.pt")
            self.position = YOLO("best_seal.pt", verbose=False)  # ===> 경로 수정 필요

            self.mp_hands = mp.solutions.hands
            self.cap = cv2.VideoCapture(0)
            print("[INFO] Camera initialized for collision detection.")
            self.initialized = True

            self.collision_thread = threading.Thread(target=self.detect_collision, daemon=True)
            self.collision_handler_thread = threading.Thread(target=self.check_collision_and_pause, daemon=True)
            

            if self.arm:
                try:
                    self.arm.motion_enable(enable=True)
                    self.arm.set_mode(0)
                    self.arm.set_state(state=0) # state : 0: sport state, 3: pause state, 4: stop state

                    print("[SUCCESS] Arm initialized successfully.")
                except Exception as e:
                    print(f"[ERROR] Failed to initialize arm: {e}")
            else:
                print("[WARNING] Arm is not connected. Skipping motion setup.")

            self.speed = 200
            self.mvacc = 100
            self.poses = [[-279.320374, -124.808502, 202.336609, -100.67006, 86.594893, 156.301798],  # 0
                      [-174.406891, -126.305305, 204.095306, -136.000585, 88.308839, 139.035772],  # 1
                      [-95.280388, -157.109482, 203.992859, -158.887098, 88.016573, 155.244748],   # 2
                      [245.419815, 133.853424, 465.297546, -26.709172, -86.311909, -60.439599],     # 3
                      [234.857269, 23.210983, 496.896393, 133.233142, -88.542319, 135.449801],        # 4
                      [30.640814, -242.950836, 475.663879, 58.721241, -90.000000, 118.777818],      # 5
                      [184.512177, -95.52092, 150.964706, 106.073968, -90.00000, -17.925959],       # 6
                      [188.249832, 150.832748, 343.810577, 8.6496, -90.000000, 66.370743],         # 7
                      [206.267776, -100.06179, 256.300354, 106.073968, -90.00000, -17.925959],    # 8
                      [9.054959, -240.938965, 331.05481, 101.944776, 90.000000, -20.444509],       # 9
                      [-226.724548, -25.836311, 330.765045, 98.07628, 90.000000, -125.882781],     # 10
                      [-245.524109, 141.818008, 338.6409, 33.684934, 90.000000, 134.253809],       # 11
                      [-169.914047, 154.82869, 336.795441, 164.204433, 90.000000, -133.145021],     # 12
                      [-3.740475, 142.690125, 318.331635, 26.406708, 90.000000, 109.23108],         # 13
                      [-285.87207, -75.438164, 297.169464, -29.021172, 90.000000, 162.908746],      # 14
                      [43.712517, -239.918655, 415.718506, 12.996803, -69.99218, 147.439917],      # 15
                      [-3.237866, -333.397522, 381.811737, 36.00398, 90.000000, -57.412663],        # 16
                      [-262.993347, -1.932795, 289.230286, -35.328005, -90.000000, 36.108488],       # 17
                      [283.96933, -2.311633, 480.578888, 123.41551, -90.000000, 75.265741],         # 18
                      [-240.917038, 53.239895, 326.868042, 47.079083, 90.000000, 135.969875],       # 19_토핑1
                      [-177.672729, 80.462181, 336.936279, 159.221247, 90.000000, -153.476944],     # 20_토핑2
                      [44.84457, 92.966362, 326.998932, -136.038229, 90.000000, -107.056878],       # 21_토핑3
                      [-285.787994, -123.837471, 297.447327, -142.491968, 89.528208, 117.708335],      # 22    
                      [-279.560455, -126.164642, 237.91156, -45.666283, -90.00000, 124.129702],    # 23
                      [250.415436, 133.272247, 468.451202, -68.57199, -90.00000, -14.233188],      # 24
                      [-174.314026, -127.809052, 277.985352, -156.920936, 88.085672, 119.191952],      # 25
                      [-226.42897, -150.633453, 224.207199, 94.248005, -90.00000, 34.899776],       # 26
                      [-97.023697, -159.977264, 262.409912, 177.465541, 87.907196, 133.068187],       # 27
                      [-79.293274, -148.774612, 250.281754, 121.364035, -90.00000, 2.294524],       # 28
                      [214.515839, 83.34877, 274.337463, 104.048162, 90.00000, -168.66405],     # 29
                      [237.438034, -47.484119, 495.882233, 174.281156, 85.629746, -108.696625],       # 30
                      [227.168167, 133.247467, 485.689301, 19.271607, 86.817602, 108.131574],       # 31
                      [-289.080017, -122.171066, 229.931335, -30.279101, 90.00000, -124.39332],       # 32
                      [-285.567535, -116.483253, 184.405716, -7.129658, 90.00000, -102.28964],    # 33
                      [-286.470978, -34.58794, 184.935486, -72.67517, 90.00000, -164.476645],     # 34
                      [-302.779175, -121.291069, 253.884583, -147.120779, -90.00000, 122.556073],     # 35
                      [-302.779175, -121.291069, 253.884583, -147.120779, -90.00000, 122.556073],     # 36
                      [-302.779175, -121.291069, 253.884583, -147.120779, -90.00000, 122.556073],     # 37
                      [-302.779175, -121.291069, 253.884583, -147.120779, -90.00000, 122.556073],     # 38
                      [-302.779175, -121.291069, 253.884583, -147.120779, -90.00000, 122.556073],     # 39
                      [-302.779175, -121.291069, 253.884583, -147.120779, -90.00000, 122.556073],     # 40
                      [-179.281235, -85.541214, 303.217468, 21.202933, 90.00000, -68.706635],          # 41
                      [-302.779175, -121.291069, 253.884583, -147.120779, -90.00000, 122.556073],     # 42
                      [103.168549, -179.884872, 456.517487, -21.966858, 70.547778, -32.688675],     # 43
                      [230.825073, 7.382851, 456.112244, -70.162464, 90.00000, 19.525198],          # 44
                      [231.686096, 134.774963, 460.133575, 32.133135, 85.384921, 121.518103],     # 45
                      [188.225983, -218.800308, 395.810577, 36.900201, -62.560803, 154.531931],     # 46
                      [-143.559464, -285.161621, 397.227142, -8.590414, -61.142733, 51.339539]]     # 47
                      
        """[-279.217865, -39.094509, 193.051575, 62.3562, 86.627552, -34.165588]
        0: 대략적 아이스크립팩 1번자리 위치. 
        1: 대략적 아이스크립팩 2번자리 위치.
        2: 대략적 아이스크림팩 3번자리 위치.
        3: 대략적인 프레스 위쪽 위치
        4: 대략적인 프레스 위쪽 바로앞 위치
        5: 대략적인 프레스 위쪽 멀리 앞 위치.
        6: 대략적인 컵 위치
        7: 대략적인 프레스 아래쪽 위치
        8: 대략적인 컵 위쪽 위치
        9: 컵 to 토핑 : 1
        10: 컵 to 토핑 : 2
        11: 대략적인 토핑1 위치
        12: 대략적인 토핑2 위치
        13: 대략적인 토핑3 위치
        14: 아이스크림 위쪽 안전위치
        15 : 14번의 위아래 반대 버전
        16: 사람
        17: 디폴트
        18: 5 - 8 중간단계
        19: 토핑 1 준비
        20: 토핑 2 준비
        21: 토핑 3 준비
        22: 아이스크림 1 준비
        23: 아이스크림 1 잡은 후
        24: 아이스크림 놓기
        25: 아이스크림 2 준비
        26: 아이스크림 2 잡은 후
        27: 아이스크림 3 준비
        28: 아이스크림 3 잡은 후
        29: 쓰레기통 위
        30: 쓰레기통-프레서 중간
        31: 프레서 쪽 집은 후
        32: 아이스크림 1 놓을 준비
        33: 아이스크림 1 놓기
        34: 아이스크림 1 놓은 후
        35: 아이스크림 2 놓을 준비
        36: 아이스크림 2 놓기
        37: 아이스크림 2 놓은 후
        38: 아이스크림 3 놓을 준비
        39: 아이스크림 3 놓기
        40: 아이스크림 3 놓은 후
        41: 아이스크림 놓을 준비
        42: 임시
        43: 프레스 위쪽 멀리 앞 위치
        44: 프레스 위쪽 바로 앞 위치
        45: 아이스크림 회수 위치
        46: 쓰레기통 to 디폴트 1
        47: 쓰레기통 to 디폴트 2
        """
        self.routes = {"get_ready": [17],
                       "default_to_ice_1": [22, 0],
                       "default_to_ice_2": [25, 1],
                       "default_to_ice_3": [27, 2],
                       "ice_1_to_up": [0, 22],
                       "ice_2_to_up": [1, 25],
                       "ice_3_to_up": [2, 27],
                       "front_press_to_in_press": [4, 3],
                       "in_press_to_front_press": [3, 24, 4],
                       "up_cup_to_cup": [8, 6],
                       "cup_to_up_cup" : [6, 8],
                       "up_cup_to_topping_zone": [9, 10, 19],  #없애고, joint control로 대체?
                       "topping_1": [19, 11],
                       "after_topping_1": [19],
                       "topping_2": [20, 12],
                       "after_topping_2": [20],
                       "topping_3": [21, 13],
                       "after_topping_3": [21],
                       "topping_to_under_press": [21, 7],
                       "under_press_to_person": [7, 21, 19, 14, 16],
                       "put_on_ice_1": [16, 41, 32, 33],
                       "put_on_ice_2": [16, 41, 35, 36],
                       "put_on_ice_3": [16, 41, 38, 39],
                       "ice_1_to_in_press_retrieve": [33, 34, 41, 43, 44, 45],
                       "ice_2_to_in_press_retrieve": [36, 37, 41, 43, 44, 45],
                       "ice_3_to_in_press_retrieve": [39, 40, 41, 43, 44, 45],
                       "person_to_press_retrieve": [16, 43, 44, 45],
                       "press_to_waste": [45, 31, 30, 29],
                       "return_to_default": [29, 30, 44, 43, 46, 47, 17], 
                       "return_to_default_direct": [46, 47, 17],
                       "just_give": [16],
                       }
        """
        컵 집고 -> 프레스아래 -> 토핑쪽으로 회전 안됨.
        아이스크림 받고. 바로 컵쪽으로 안됨.(무조건 토핑쪽 방향 이용)
        """
        self.angles = {"ice_to_front_press": [-9.3, -0.3, 81.8, -92.1, -97.8, 80.8],  # 각 angle 값 대입
                        "front_press_to_up_cup": [-10.5, 14.1, 35.2, 81.9, -86.9, -20.9],
                        "up_cup_to topping_zone":[-165, 8.5, 48.8, 86.3, -82.3, 142.5],
                        }
    @staticmethod
    def get_instance():
        """ Flask 스레드 환경에서도 싱글톤 유지 """
        if 'arm_instance' not in g:
            g.arm_instance = A_Circle_Arm("192.168.1.182", app)
        return g.arm_instance

    def connect_to_arm(self, max_retries=5, retry_delay=1):
        """ Try connecting to the robotic arm with retry logic. """
        attempt = 0
        while attempt < max_retries:
            try:
                print(f"[INFO] Attempting to connect to XArm at {self.arm_ip} (Attempt {attempt + 1}/{max_retries})")
                self.arm = XArmAPI(self.arm_ip)
                
                if self.arm.connected:
                    print(f"[SUCCESS] Connected to XArm at {self.arm_ip}")
                    return
                else:
                    print(f"[WARNING] Connection established, but arm is not fully functional.")

            except Exception as e:
                print(f"[ERROR] Failed to connect to XArm at {self.arm_ip}: {e}")

            attempt += 1
            time.sleep(retry_delay)
        print(f"[CRITICAL] Unable to connect to XArm at {self.arm_ip} after {max_retries} attempts. Proceeding without arm control.")


    def detect_sealing_location(self):
        """YOLO 모델을 사용하여 실링을 감지하고 가장 가까운 아이스크림 위치 번호를 반환"""
        print("[INFO] 실링 감지 시작... (실링이 감지될 때까지 계속 실행)")

        if not self.cap.isOpened():
            print("[ERROR] 기존 카메라(self.cap)가 열려 있지 않음!")
            return None  # 기존 카메라가 닫혀 있으면 실행할 수 없음

        while self.running:
            ret, frame = self.cap.read()  
            if not ret:
                print("[ERROR] 카메라에서 프레임을 읽을 수 없습니다!")
                break

            frame_copy = frame.copy()
            results = self.position(frame_copy)  # YOLO 모델 실행
            #cv2.imshow("Sealing Check", frame)
            

            # 🔹 YOLO 결과를 올바르게 확인하는 코드
            if results and hasattr(results[0], "boxes") and len(results[0].boxes) > 0:
                
                for result in results:
                    
                    if result.boxes.cls is not None:
                        
                        for i, box in enumerate(result.boxes.xyxy):
                            class_id = int(result.boxes.cls[i])
                            class_name = self.position.names.get(class_id, "Unknown")
                            print(class_name)

                            if class_name == "sealing":  # 실링 객체 감지됨
                                x1, y1, x2, y2 = map(int, box)
                                center_x = (x1 + x2) / 2
                                center_y = (y1 + y2) / 2
                                print(f"[INFO] 실링 감지됨: ({center_x}, {center_y})")

                                # 가장 가까운 아이스크림 위치 찾기
                                self.position = self._find_closest_ice(center_x, center_y)
                                print(f"[INFO] 가장 가까운 아이스크림 위치: {self.position}")

                                # 실링 감지가 완료되면 종료
                                self.running = False
                                #cv2.destroyWindow("Sealing Check")
                                self.cap.release()
                                # cv2.waitKey(1)
                                print("[INFO] 실링 감지 완료.")
                                # 충돌 감지 관련 멀티스레드
                                
                                return self.position  # 1, 2, 3 중 하나 반환
                                
            # 'q' 키를 누르면 종료
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        self.cap.release()
        # cv2.waitKey(1)
            

    def _find_closest_ice(self, x, y):
        """실링 감지된 좌표와 가장 가까운 아이스크림 위치 번호 반환"""
        ice_positions = {
            1: (494, 99),  # Ice 1
            2: (410, 100),  # Ice 2
            3: (324, 101)  # Ice 3
        }
        closest_ice = min(ice_positions.keys(), key=lambda k: np.linalg.norm(np.array([x, y]) - np.array(ice_positions[k])))
        return closest_ice


    def check_collision_and_pause(self):
        """충돌이 감지되면 로봇을 멈추고, 충돌이 해제될 때까지 대기"""
        while 1:
            if self.collision_detected:
                print("[WARNING] Collision detected! Pausing motion")
                self.arm.set_state(state=3)  # 3: Pause state (정지)
                time.sleep(2)
                
            else:
                #print("[INFO] Collision cleared. Resuming motion")
                self.arm.set_state(state=0)  # 0: Resume motion (재개)

            time.sleep(0.1)

    def stop_collision_thread(self):
        if self.collision_thread.is_alive() and self.collision_thread : 
            self.collision_thread.join()
            self.collision_handler_thread.join()
    def detect_collision(self):
        """손과 로봇팔의 충돌 감지를 수행"""
        self.last_no_collision_time = None  # 최근 충돌이 없었던 시간을 기록
        print("detect_collision")

        if not self.cap.isOpened():
            print("[ERROR] 기존 카메라(self.cap)가 열려 있지 않음!")
            self.cap = cv2.VideoCapture(0)
            return None  # 기존 카메라가 닫혀 있으면 실행할 수 없음


        with self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("[ERROR] 기존 카메라에서 프레임을 읽을 수 없음!")
                    break

                results = self.model(frame, task="segment", conf=0.25, verbose=False)
                robot_masks = []
                for result in results:
                    if result.masks is not None:
                        for mask in result.masks.xy:
                            mask = np.array(mask, dtype=np.int32)
                            robot_masks.append(mask)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_results = hands.process(rgb_frame)

                collision_detected_now = False  # 현재 프레임에서 충돌 감지 여부

                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        for idx, landmark in enumerate(hand_landmarks.landmark):
                            h, w, _ = frame.shape
                            hand_x, hand_y = int(landmark.x * w), int(landmark.y * h)

                            # 🔹 충돌 감지 확인
                            for mask in robot_masks:
                                if cv2.pointPolygonTest(mask, (hand_x, hand_y), False) >= 0:
                                    collision_detected_now = True  # 이번 프레임에서 충돌 발생
                                    break  # 한 번이라도 충돌 감지되면 즉시 탈출

                # 🔹 충돌이 감지된 경우
                if collision_detected_now:
                    if not self.collision_detected:  # 새롭게 충돌 감지가 되었을 때만 출력
                        print("[ALERT] Collision detected!")
                    self.collision_detected = True
                    self.last_no_collision_time = None  # 충돌이 감지되면 타이머 초기화

                # 🔹 손이 감지되지 않거나 충돌이 없을 경우
                else:
                    # print("Keep going")
                    if self.last_no_collision_time is None:  
                        self.last_no_collision_time = time.time()  # 최초 충돌이 없는 순간 기록

                    elif time.time() - self.last_no_collision_time >= 1.0:  
                        self.collision_detected = False  # 1초 동안 충돌이 없으면 False로 변경


                # 🔹 충돌이 감지되었을 경우 로봇을 멈춤
                if self.collision_detected:
                    cv2.putText(frame, "Collision Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 0, 255), 3, cv2.LINE_AA)  # 빨간색 텍스트 출력



                # cv2.imshow("Robot Arm & Hand Tracking", frame)

                # # 'q' 키를 누르면 종료
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break      
    def threading_start(self):
        print("[INFO] Threading thying start!")
        if not self.collision_thread.is_alive() and not self.collision_handler_thread.is_alive() and not self.stop_collision_thread_status :  
            # self.collision_thread = threading.Thread(target=self.detect_collision)
            self.collision_thread.start()
            self.collision_handler_thread.start()
            print("[INFO] Collision Thread Started!")
            print("[INFO] Collision Thread Started!")
            print("[INFO] Collision Thread Started!")
            print("[INFO] Collision Thread Started!")
            print("[INFO] Collision Thread Started!")
            print("[INFO] Collision Thread Started!")
            print("[INFO] Collision Thread Started!")
            print("[INFO] Collision Thread Started!")
            print("[INFO] Collision Thread Started!")
            print("[INFO] Collision Thread Started!")
            self.stop_collision_thread_status = True
            self.stop_collision_handler_thread_status = True

        
    def set_collision_status(self, status): # 현재 사용 x
        self.collision_detected = status
        if status:
            print("[ALERT] Collision detected!")
        else:
            print("[INFO] Collision cleared.")

    def _return_to_default(self):
        joint_angles = [-180, -43.4, 23, -7, -24, 4.8]
        for i, angle in enumerate(joint_angles, start=1):
            self.arm.set_servo_angle(servo_id=i, angle=angle, speed=self.speed, mvacc=self.mvacc, wait=True)

    def _init_6th_motor(self):
        self.arm.set_servo_angle(servo_id=6, angle=0, speed=self.speed, mvacc=self.mvacc, relative=False, wait=True)

    def _grap(self, gripper=True):
        if gripper:
            self.arm.close_lite6_gripper() 
            time.sleep(1)
        else:
            self.arm.open_lite6_gripper()
            time.sleep(1)
            self.arm.stop_lite6_gripper()

    def _move_one_path(self, act, pitch_maintain=True):
        """
        act : self.routes의 key 중 하나
        pitch_maintain = 이전 움직임의 pitch를 유지할지 여부
        """
        if act not in self.routes:
            print(f"잘못된 경로: {act}")
            return
        
        route = self.routes[act]
        for pose_index in route:
            pose = self.poses[pose_index]
            
            if pitch_maintain:
                pre_pitch = self.arm.get_position()
                pre_pitch = pre_pitch[1][4]
                pose[4] = pre_pitch  # 기존 pitch 유지
            
            """while self.collision_detected:
                print(f"[WARNING] Collision detected while moving to pose {pose_index}. Pausing...")
                self.check_collision_and_pause()"""
            
            print(f"[INFO] Moving to pose {pose_index}: {pose}")
            self.arm.set_position(*pose, speed=self.speed, mvacc=self.mvacc, wait=True)


        print(f"[INFO] Path '{act}' completed.")


    def _move_joint_angle(self, act):
        """
        Joint Control 기반으로 경로를 이동하는 함수
        act : self.joint_routes의 key 중 하나[-289.172333, -60.808853, 205.619354, 112.430763, -90.00000, -25.71744]
        """
        if act not in self.angles:
            print(f"[ERROR] 잘못된 Joint 경로: {act}")
            return

        joint_angles = self.angles[act]  # 해당 경로의 Joint Angles 가져오기
        
        print(f"[INFO] Moving to angle '{act}': {joint_angles}")

        # ✅ Joint Control 적용 (모든 서보모터 동시에 이동)
        self.arm.set_servo_angle(angle=joint_angles, speed=self.speed - 20, mvacc=self.mvacc, wait=True)

        print(f"[INFO] Angle '{act}' completed.")


    def _turn_cup(self, angle):
        # 6번 모터 +360 ~ -360 까지.
        cur_6_motor_angle = self.arm.get_servo_angle(servo_id=6)
        angle = abs(angle)
        if cur_6_motor_angle[1] > 0:
            angle = -angle
        self.arm.set_servo_angle(servo_id=6, angle=angle, speed=self.speed, mvacc=self.mvacc, relative=True, wait=True)
    
    def select_toppings(self):
        """ Flask route handler: receives topping data & moves the robot """
        if request.method == "OPTIONS":
            # Return an empty response with 200 OK status.
            return ('', 200)

        data = request.get_json()  # ✅ Use .get_json() to avoid errors
        if not data:
            return self.create_response({"error": "No data received"}, 400)
            print("[INFO] Received data:",aol==p)

        toppings = data.get('toppings', [])
        print("[INFO] Received Toppings:", toppings)

        # ✅ Only process movement if the arm is connected
        if self.arm:
            threading.Thread(target=self.run, args=(toppings,), daemon=True).start()

            # self.run(toppings)
        else:
            print("[WARNING] Robot arm is not connected. Skipping movement.")

        return jsonify({
            "message": "Toppings received and processed",
            "received_toppings": toppings
        }), 200


    def create_response(self, data, status=200):
        """ ✅ Ensure all responses contain CORS headers """    
        response = jsonify(data)
        response.status_code = status
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        return response

    
    def run(self, toppings):

        """전체 프로세스 실행"""
        topping_1 = toppings[0]
        topping_2 = toppings[1]
        topping_3 = toppings[2]

        print("[INFO] 프로그램 시작")

        # 실링 감지 시작 (실링이 감지될 때까지 무한 반복)
        detected_position = self.detect_sealing_location()
        self.cap.release()
        self.cap = None
        self.cap = cv2.VideoCapture(0)
        # cv2.waitKey(1)

        # 아이스크림 위치 번호 출력 (추후 동작 제어는 따로 처리)
        print(f"[INFO] 최종 반환된 위치 번호: {detected_position}")

        time.sleep(2)
        

        self.threading_start()




        # 토핑 선택과 관련된 초기 설정
        self._init_6th_motor()  # 6번째 모터 초기화
        self._return_to_default()
        self._grap(False)  # 그랩 초기화
        self._move_one_path("get_ready", pitch_maintain=False)  # 기존 초기위치
        self._turn_cup(180)

        if detected_position == 1:
            self._move_one_path("default_to_ice_1", pitch_maintain=False)  # 기본 경로로 이동
            self._grap(True)  # 아이스크림 그랩
            self._move_one_path("ice_1_to_up")  

        if detected_position == 2:
            self._move_one_path("default_to_ice_2", pitch_maintain=False)  # 기본 경로로 이동
            self._grap(True)  # 아이스크림 그랩
            self._move_one_path("ice_2_to_up")  

        if detected_position == 3:
            self._move_one_path("default_to_ice_3", pitch_maintain=False)  # 기본 경로로 이동
            self._grap(True)  # 아이스크림 그랩
            self._move_one_path("ice_3_to_up")

        self._turn_cup(-180)
        self._move_joint_angle("ice_to_front_press")
        self._move_one_path("front_press_to_in_press")  # 아이스크림 프레스 이동
        self._grap(False)  # 그랩 해제
        self._move_one_path("in_press_to_front_press")  # 프레스로 이동
        self._move_joint_angle("front_press_to_up_cup")
        self._move_one_path("up_cup_to_cup")  # 컵 위로 이동 
        self._grap(True)  # 컵 그랩
        self._move_one_path("cup_to_up_cup")  # 컵 위로 이동
        self._turn_cup(180)  # 컵 회전
        self._grap(False)  # 그랩 해제

        self._move_one_path("up_cup_to_topping_zone")

        # 선택된 토핑에 따라 동작 수행
       
        if topping_1 and topping_2 and topping_3:
            self._move_one_path("topping_1")
            time.sleep(0.2)
            self.arm.set_cgpio_digital(0,1)
            time.sleep(3)
            self.arm.set_cgpio_digital(0,0)
            time.sleep(0.2)
            self._move_one_path("after_topping_1")
            self._move_one_path("topping_2")
            time.sleep(0.2)
            self.arm.set_cgpio_digital(1,1)
            time.sleep(3)
            self.arm.set_cgpio_digital(1,0)
            time.sleep(0.2)
            self._move_one_path("after_topping_2")
            self._move_one_path("topping_3")
            time.sleep(0.2)
            self.arm.set_cgpio_digital(2,1)
            time.sleep(3)
            self.arm.set_cgpio_digital(2,0)
            time.sleep(0.2)
            self._move_one_path("after_topping_3")

        elif topping_1 and topping_2:
            self._move_one_path("topping_1")
            time.sleep(0.2)
            self.arm.set_cgpio_digital(0,1)
            time.sleep(3)
            self.arm.set_cgpio_digital(0,0)
            time.sleep(0.2)
            self._move_one_path("after_topping_1")
            self._move_one_path("topping_2")
            time.sleep(0.2)
            self.arm.set_cgpio_digital(1,1)
            time.sleep(3)
            self.arm.set_cgpio_digital(1,0)
            time.sleep(0.2)
            self._move_one_path("after_topping_2")

        elif topping_1 and topping_3:
            self._move_one_path("topping_1")
            time.sleep(0.2)
            self.arm.set_cgpio_digital(0,1)
            time.sleep(3)
            self.arm.set_cgpio_digital(0,0)
            time.sleep(0.2)
            self._move_one_path("after_topping_1")
            self._move_one_path("topping_3")
            time.sleep(0.2)
            self.arm.set_cgpio_digital(2,1)
            time.sleep(3)
            self.arm.set_cgpio_digital(2,0)
            time.sleep(0.2)
            self._move_one_path("after_topping_3")
        
        elif topping_2 and topping_3:
            self._move_one_path("topping_2")
            time.sleep(0.2)
            self.arm.set_cgpio_digital(1,1)
            time.sleep(3)
            self.arm.set_cgpio_digital(1,0)
            time.sleep(0.2)
            self._move_one_path("after_topping_2")
            self._move_one_path("topping_3")
            time.sleep(0.2)
            self.arm.set_cgpio_digital(2,1)
            time.sleep(3)
            self.arm.set_cgpio_digital(2,0)
            time.sleep(0.2)
            self._move_one_path("after_topping_3")

        elif topping_1:
            self._move_one_path("topping_1")
            time.sleep(0.2)
            self.arm.set_cgpio_digital(0,1)
            time.sleep(3)
            self.arm.set_cgpio_digital(0,0)
            time.sleep(0.2)
            self._move_one_path("after_topping_1")

        elif topping_2:
            self._move_one_path("topping_2")
            time.sleep(0.2)
            self.arm.set_cgpio_digital(1,1)
            time.sleep(3)
            self.arm.set_cgpio_digital(1,0)
            time.sleep(0.2)
            self._move_one_path("after_topping_2")
        
        elif topping_3:
            self._move_one_path("topping_3")
            time.sleep(0.2)
            self.arm.set_cgpio_digital(2,1)
            time.sleep(3)
            self.arm.set_cgpio_digital(2,0)
            time.sleep(0.2)
            self._move_one_path("after_topping_3")
        
        else:
            self._move_one_path("after_topping_3")
            
        
        # 모든 토핑이 완료되었으면, 후속 동작 실행
        self._move_one_path("topping_to_under_press")  # 후속 동작 이동
        self.arm.set_cgpio_digital(3,1)
        time.sleep(12)
        self.arm.set_cgpio_digital(3,0) 
        
        
        self._move_one_path("under_press_to_person")  # 사람에게 전달

        ############# -> 나가기 대충 5초전 플래그 여기 걸기 여기
        ############# -> 나가기 대충 5초전 플래그 여기 걸기 여기
        ############# -> 나가기 대충 5초전 플래그 여기 걸기 여기
        ############# -> 나가기 대충 5초전 플래그 여기 걸기 여기
        ############# -> 나가기 대충 5초전 플래그 여기 걸기 여기
        ############# -> 나가기 대충 5초전 플래그 여기 걸기 여기
        self.end_check_point = True  
        ############# -> 나가기 대충 5초전 플래그 여기 걸기 여기
        ############# -> 나가기 대충 5초전 플래그 여기 걸기 여기
        ############# -> 나가기 대충 5초전 플래그 여기 걸기 여기
        ############# -> 나가기 대충 5초전 플래그 여기 걸기 여기
        ############# -> 나가기 대충 5초전 플래그 여기 걸기 여기
        ############# -> 나가기 대충 5초전 플래그 여기 걸기 여기


        time.sleep(5)  # 잠시 대기
        self._move_one_path("just_give")  # 아이스크림 전달
        

        start_time = time.time()

        while time.time() - start_time < 30:
            try:
                response = requests.get('http://control_service:8080/check_ice_cream_status', timeout=5)
                if response.json()['ice_cream_taken']:
                    self.ice_cream_taken = True
                    break
            except Exception as e:
                print(f"Error checking ice cream status: {e}")
            time.sleep(0.5)
            print("waiting for ice cream taken....", flush=True)
            
            if self.ice_cream_taken:   # 아이스크림을 가져갔다면 ====> 여기서 '아이스크림을 사람이 가져갔다' 라는 정보가 입력되어야 하는데, 어떤 식으로 구현해야 할지 모르겠습니다..
                #self._move_one_path("person_to_press_retrieve") # 바로 프레스로 이동
                self._grap(False)  # 그랩 해제
                break
            
        else:      # 아이스크림을 안 가져갔다면
            print("아이스크림을 안 가져갔다면", flush=True)
            #self._move_one_path("put_on_ice_1")  # 아이스크림 위치에 올리기
            #self._move_one_path("ice_1_to_in_press_retrieve")  # 그 후 프레스로 이동
            self._grap(False)  # 그랩 해제
        
        self._grap(True)  # 다시 그랩
        print("아이스크림 버리는 위치로 이동", flush=True)
        #self._move_one_path("press_to_waste")  # 아이스크림 버리는 위치로
        self._turn_cup(-180)  # 컵 회전
        self._grap(False)  # 그랩 해제
        """time.sleep(3)  # 3초 대기"""
        self._return_to_default() # 기본 위치로 돌아가기

        """self._move_one_path("return_to_default")"""
        """self.arm.set_cgpio_analog(0, 5)
        time.sleep(3)
        self.arm.set_cgpio_analog(1, 5)
        time.sleep(3)
        self.arm.set_cgpio_analog(0, 0)
        time.sleep(3)"""


my_arm = A_Circle_Arm("192.168.1.182", app)

@app.route('/ice_cream_taken', methods=['POST'])
def ice_cream_taken_handler():
    arm = A_Circle_Arm.get_instance() # 싱글톤 인스턴스 가져오기
    data = request.get_json()
    print(data)
    if data['status'] == "taken":
        arm.ice_cream_taken = True
        print("Ice cream taken status received")
        return jsonify({"message": "Ice cream taken status received"}), 200
    else:
        arm.ice_cream_taken = False
        print("Ice cream not taken status received")
        return jsonify({"message": "XXXXX Ice cream not taken status received"}), 200
@app.route('/check_ice_cream_status', methods=['GET'])
def check_ice_cream_status():
    arm = A_Circle_Arm.get_instance() # 싱글톤 인스턴스 가져오기
    return jsonify({"ice_cream_taken": arm.ice_cream_taken}), 200

#메모리페이지한테 제조 상태 알려주기 
@app.route('/check_end_status', methods=['GET'])
def check_end_status():
    try:
        arm = A_Circle_Arm.get_instance()  # 싱글톤 인스턴스 가져오기
        if arm.end_check_point:
            return jsonify({"status": "end_ice"}), 200
        else:
            return jsonify({"status": "processing"}), 200
    except Exception as e:
        print(f"🔥 /check_end_status 오류 발생: {e}", flush=True)
        return jsonify({"error": str(e)}), 500
       

if __name__ == "__main__":
    # Run Flask server
    app.run(host='0.0.0.0', port=8080, threaded=True)