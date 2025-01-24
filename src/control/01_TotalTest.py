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
                      [-170.092712, -121.410614, 198.670761, -92.245346, -90.000000, -177.668826],  # 1
                      [-89.333214, -146.576248, 200.222748, -85.792695, -90.000000, -147.795666],   # 2
                      [252.956329, 109.117012, 482.582855, 84.924435, -90.000000, -167.982402],     # 3
                      [237.472427, 11.561559, 469.93689, 103.99946, -90.000000, 165.818741],        # 4
                      [30.640814, -242.950836, 475.663879, 58.721241, -90.000000, 118.777818],      # 5
                      [157.819351, -103.05304, 151.674011, -87.493234, -90.000000, -172.962666],    # 6
                      [188.249832, 150.832748, 343.810577, 8.6496, -90.000000, 66.370743],     # 7
                      [166.209717, -86.796677, 235.458038, -87.493234, -90.000000, -172.962666],    # 8
                      [9.054959, -240.938965, 331.05481, 101.944776, 90.000000, -20.444509],       # 9
                      [-226.724548, -25.836311, 330.765045, 98.07628, 90.000000, -125.882781],     # 10
                      [-245.524109, 141.818008, 338.6409, 33.684934, 90.000000, 134.253809],       # 11
                      [-169.914047, 154.82869, 336.795441, 164.204433, 90.000000, -133.145021],     # 12
                      [-3.740475, 142.690125, 318.331635, 26.406708, 90.000000, 109.23108],      # 13
                      [-285.87207, -75.438164, 297.169464, -29.021172, 90.000000, 162.908746],      # 14
                      [-116.726761, -321.590057, 358.78952, 52.845158, -90.000000, 15.316365],      # 15
                      [17.602173, -355.825043, 393.381561, 59.251915, 90.00000, -30.14944],    # 16
                      [-295.006042, -119.118347, 341.861694, 60.958527, -90.000000, 25.42059],       # 17
                      [283.96933, -2.311633, 480.578888, 123.41551, -90.000000, 75.265741],         # 18
                      [-240.917038, 53.239895, 326.868042, 47.079083, 90.000000, 135.969875],       # 19_토핑1
                      [-177.672729, 80.462181, 336.936279, 159.221247, 90.000000, -153.476944],     # 20_토핑2
                      [44.84457, 92.966362, 326.998932, -136.038229, 90.000000, -107.056878],       # 21_토핑3
                      [-289.172333, -60.808853, 205.619354, 112.430763, -90.00000, -25.71744],
                      [-279.560455, -126.164642, 237.91156, -45.666283, -88.686304, 124.129702]]
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
        16: 사람.
        17: 디폴트
        18: 5 - 8 중간단계
        """
        self.routes = {"default_to_ice_1": [17, 22, 0, 23],
                       "ice_1_to_in_press": [23, 15, 5, 4, 3],
                       "ice_2_to_in_press": [1, 15, 5 ,4, 3],
                       "ice_3_to_in_press": [2, 15, 5, 4, 3],
                       "in_press_to_cup": [3, 4, 5, 18, 8, 6],
                       "cup_to_up_cup" : [6, 8],
                       "up_cup_to_topping_1": [9, 10, 19, 11],  # 10에서 11가면서 컵 방향 회전.
                       "topping_1_to_topping_2": [11, 19, 12],
                       "topping_2_to_topping_3": [12, 20, 13],
                       "toping_3_to_under_press": [13, 21, 7],
                       "under_press_to_person": [7, 21, 19, 14, 16], # 14는 디폴트 정해서수정필요
                       "person_to_default": [16, 17], # 14에서 디폴트로 추가 필요.
                       }
        """
        컵 집고 -> 프레스아래 -> 토핑쪽으로 회전 안됨.
        아이스크림 받고. 바로 컵쪽으로 안됨.(무조건 토핑쪽 방향 이용)
        """
    def move_a_point(self, num):
        ret = self.arm.set_position(*self.poses[num], speed=self.speed, mvacc=self.mvacc, wait=True)

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
        time.sleep(30)
        self._move_one_path("under_press_to_person")
        # self._move_one_path("person_to_default")

if __name__ == "__main__":
    my_arm = A_Circle_Arm("192.168.1.182")
    # my_arm._move_one_path("cup_to_front_topping")
    # my_arm.move_a_point(17)
    my_arm.run()
    # my_arm._control_last_joint(90)
    # my_arm._grap(True) 