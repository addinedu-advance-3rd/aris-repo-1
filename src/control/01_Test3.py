#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2019, UFACTORY, Inc.
# All rights reserved.
#
# Author: Vinman <vinman.wen@ufactory.cc> <vinman.cub@gmail.com>

"""
Description: Move Circle
"""

import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI


#######################################################
# """
# Just for test example
# """
# if len(sys.argv) >= 2:
#     ip = sys.argv[1]
# else:
#     try:
#         from configparser import ConfigParser
#         parser = ConfigParser()
#         parser.read('../robot.conf')
#         ip = parser.get('xArm', 'ip')
#     except:
#         ip = input('Please input the xArm ip address:')
#         if not ip:
#             print('input error, exit')
#             sys.exit(1)
########################################################
class A_Circle_Arm():
    def __init__(self, arm_ip):
        self.arm = XArmAPI(arm_ip)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0) # state : 0: sport state, 3: pause state, 4: stop state
        self.speed = 200
        self.mvacc = 100
        self.poses = [[-292.814117, -127.788345, 210.566269, 122.390603, -90.000000, -38.036663],  # 0
                      [-197.424606, -130.498032, 205.028625, 25.661519, -90.000000, 82.715682],  # 1
                      [-73.66082, -146.904449, 206.089279, 89.435503, -90.000000, 31.025493],   # 2
                      [248.424866, 133.485321, 489.891785, 144.414986, -90.00000, 130.577884],     # 3
                      [234.188812, 29.031912, 507.134094, 82.900805, -90.000000, -174.530393],        # 4
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
                      [-289.172333, -60.808853, 205.619354, 112.430763, -90.00000, -25.71744],      # 22    
                      [-279.560455, -126.164642, 237.91156, -45.666283, -90.00000, 124.129702],    # 23
                      [250.415436, 133.272247, 468.451202, -68.57199, -90.00000, -14.233188],      # 24
                      [-211.21051, -78.423111, 209.862823, 169.779312, -90.00000, -62.88504],      # 25
                      [-226.42897, -150.633453, 224.207199, 94.248005, -90.00000, 34.899776],       # 26
                      [-109.640892, -81.495331, 210.433609, 128.158569, -90.00000, -6.42681],       # 27
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
                      
        """
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
        self.routes = {"default_to_ice_1": [17, 22, 0],
                       "default_to_ice_2": [17, 25, 1],
                       "default_to_ice_3": [17, 27, 2],
                       "ice_1_to_in_press": [0, 23, 15, 5, 4, 3],
                       "ice_2_to_in_press": [1, 26, 5 ,4, 3],
                       "ice_3_to_in_press": [2, 28, 5, 4, 3],
                       "in_press_to_cup": [3, 24, 4, 5, 18, 8, 6],
                       "cup_to_up_cup" : [6, 8],
                       "up_cup_to_topping_1": [9, 10, 19, 11],  # 10에서 11가면서 컵 방향 회전.
                       "topping_1_to_topping_2": [11, 19, 12],
                       "topping_2_to_topping_3": [12, 20, 13],
                       "toping_3_to_under_press": [13, 21, 7],
                       "under_press_to_person": [7, 21, 19, 14, 16], # 14는 디폴트 정해서수정필요
                       "put_on_ice_1": [16, 41, 32, 33],
                       "put_on_ice_2": [16, 41, 35, 36],
                       "put_on_ice_3": [16, 41, 38, 39],
                       "ice_1_to_in_press_retrieve": [33, 34, 41, 43, 44, 45],
                       "ice_2_to_in_press_retrieve": [36, 37, 41, 15, 5, 4, 24],
                       "ice_3_to_in_press_retrieve": [39, 40, 41, 15, 5, 4, 24],
                       "press_to_waste": [45, 31, 30, 29],
                       "return_to_default": [29, 30, 44, 43, 46, 47, 17], # 14에서 디폴트로 추가 필요.
                       "return_to_default_direct": [46, 47, 17],
                       "just_give": [16],
                       }
        """
        컵 집고 -> 프레스아래 -> 토핑쪽으로 회전 안됨.
        아이스크림 받고. 바로 컵쪽으로 안됨.(무조건 토핑쪽 방향 이용)
        """
    def move_a_point(self, num):
        if num == 17:  # 디폴트 위치만 Joint Control 사용
            joint_angles = [-179.7, 3.6, 33.5, -0.9, -60.1, 0.4]  # 예제: self.poses[17]에 맞게 수정 필요
            for i, angle in enumerate(joint_angles, start=1):
                self.arm.set_servo_angle(servo_id=i, angle=angle, speed=self.speed, mvacc=self.mvacc, wait=True)
        else:
            self.arm.set_position(*self.poses[num], speed=self.speed, mvacc=self.mvacc, wait=True)


    def _move_one_path(self, act, pitch_maintain=True):
        """
        act : self.route key 중에 하나일것
        pitch_maintain = 직전움직임의 pitch를 유지하면서 움직이는게 안전할듯하여 추가.
        """
        route = self.routes[act]
        for r_n in route:
            if pitch_maintain:
                pre_pitch = self.arm.get_position()
                pre_pitch = pre_pitch[1][4]
                pose = self.poses[r_n]
                pose [4] = pre_pitch
            else:
                pose = self.poses[r_n]
            self.arm.set_position(*pose, speed=self.speed, mvacc=self.mvacc, wait=True)

    def _grap(self, gripper=True):
        if gripper:
            self.arm.close_lite6_gripper() # close 한 상태로 유지해야만 잡고있는지 ? 테스트
            time.sleep(1)
        else:
            self.arm.open_lite6_gripper()
            time.sleep(1)
            self.arm.stop_lite6_gripper()

    def _turn_cup(self, angle):
        # 6번 모터 +360 ~ -360 까지.
        cur_6_motor_angle = self.arm.get_servo_angle(servo_id=6)
        angle = abs(angle)
        if cur_6_motor_angle[1] > 0:
            angle = -angle
        self.arm.set_servo_angle(servo_id=6, angle=angle, speed=self.speed, mvacc=self.mvacc, relative=True, wait=True)

    def _init_6th_motor(self):
        self.arm.set_servo_angle(servo_id=6, angle=0, speed=self.speed, mvacc=self.mvacc, relative=False, wait=True)

    def run(self):
        self._init_6th_motor() # 수정을하다보면 6번 모터가 360도를 넘어버리는 상황 방지
        self._grap(False)
        self._move_one_path("default_to_ice_1", pitch_maintain=False)
        self._grap(True)
        self._move_one_path("ice_1_to_in_press")
        self._grap(False)
        self._move_one_path("in_press_to_cup")
        self._grap(True)
        self._move_one_path("cup_to_up_cup")
        self._turn_cup(180)
        self._grap(False)
        self._move_one_path("up_cup_to_topping_1")
        self._move_one_path("topping_1_to_topping_2")
        self._move_one_path("topping_2_to_topping_3")
        self._move_one_path("toping_3_to_under_press")
        time.sleep(5)
        self._move_one_path("under_press_to_person")
        time.sleep(5)
        self._move_one_path("just_give")
        self._move_one_path("put_on_ice_1")
        self._move_one_path("ice_1_to_in_press_retrieve")
        self._grap(True)
        self._move_one_path("press_to_waste")
        self._turn_cup(-180)
        self._grap(False)
        time.sleep(3)
        self._move_one_path("return_to_default")
        """self._move_one_path("return_to_default_direct")"""

if __name__ == "__main__":
    my_arm = A_Circle_Arm("192.168.1.182")
    # my_arm._move_one_path("cup_to_front_topping")
    # my_arm.move_a_point(17)
    my_arm.run()
    # my_arm._control_last_joint(90)
    # my_arm._grap(True) 