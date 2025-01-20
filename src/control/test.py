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
        self.poses = [[-303.092712, -121.410614, 198.670761, -92.245346, -87.257748, -177.668826],
                      [-170.092712, -121.410614, 198.670761, -92.245346, -87.257748, -177.668826],
                      [-89.333214, -146.576248, 200.222748, -85.792695, -86.842755, -147.795666],
                      [252.956329, 109.117012, 482.582855, 84.924435, -82.6259, -167.982402],
                      [237.472427, 11.561559, 469.93689, 103.99946, -82.608024, 165.818741],
                      [30.640814, -242.950836, 475.663879, 58.721241, -82.001319, 118.777818],
                      [156.644592, -94.609383, 159.885483, -98.206628, -84.13444, -153.788805],
                      [251.656921, 138.446381, 323.958588, 72.693046, -86.16357, -167.141185],
                      [166.209717, -86.796677, 202.458038, -87.493234, -81.706016, -172.962666],
                      [9.054959, -240.938965, 331.05481, 101.944776, -80.831167, -20.444509],
                      [-226.724548, -25.836311, 330.765045, 98.07628, -72.202537, -125.882781],
                      [-206.165802, 149.329803, 340.970917, -101.214713, -82.057354, 44.894566],
                      [-121.567757, 154.914078, 339.926971, -66.56469, -83.895746, -20.461813],
                      [-16.239532, 143.714157, 336.908478, -103.106907, -88.539053, 6.292853],
                      [-116.726761, -321.590057, 358.78952, 52.845158, -75.035813, 15.316365],
                      [-95.262672, -373.135468, 286.075012, -78.853374, -86.798981, 173.378805],
                      [-48.123817, -197.722977, 282.662964, 25.46362, -84.609824, 60.776097]]
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
        15: 사람.
        16: 디폴트
        """
        self.routes = {"default_to_ice_1": [16, 0],
                       "ice_1_to_in_press": [0, 14, 5, 4, 3],
                       "ice_2_to_in_press": [1, 14, 5 ,4, 3],
                       "ice_3_to_in_press": [2, 14, 5, 4, 3],
                       "in_press_to_cup": [3, 4, 5, 8, 6],
                       "cup_to_topping_1": [6, 8, 9, 10, 11],
                       "topping_1_to_topping_2": [11, 12],
                       "topping_2_to_topping_3": [12, 13],
                       "toping_3_to_under_press": [13, 7],
                       "under_press_to_person": [7, 13, 11, 14, 15], # 14는 디폴트 정해서수정필요
                       "person_to_default": [15, 16], # 14에서 디폴트로 추가 필요.
                       }
        """
        컵 집고 -> 프레스아래 -> 토핑쪽으로 회전 안됨.
        아이스크림 받고. 바로 컵쪽으로 안됨.(무조건 토핑쪽 방향 이용)

        TODO : 5에서 8로 이동할때 충돌 일어남.
        """
    def move_a_point(self, num):
        ret = self.arm.set_position(*self.poses[num], speed=self.speed, mvacc=self.mvacc, wait=True)

    def _move_one_path(self, act):
        """
        act : self.route key 중에 하나일것
        """
        route = self.routes[act]
        for r_n in route:
            pose = self.poses[r_n]
            self.arm.set_position(*pose, speed=self.speed, mvacc=self.mvacc, wait=True)

    def _grap(self, gripper=True):
        if gripper:
            self.arm.close_lite6_gripper()
            time.sleep(1)
            self.arm.stop_lite6_gripper()
        else:
            self.arm.open_lite6_gripper()
            time.sleep(1)
            self.arm.stop_lite6_gripper()
            
    def run(self):
        self._move_one_path("default_to_ice_1")
        self._move_one_path("ice_1_to_in_press")
        self._move_one_path("in_press_to_cup")
        self._move_one_path("cup_to_topping_1")
        self._move_one_path("topping_1_to_topping_2")
        self._move_one_path("topping_2_to_topping_3")
        self._move_one_path("toping_3_to_under_press")
        self._move_one_path("under_press_to_person")
        self._move_one_path("person_to_default")

if __name__ == "__main__":
    my_arm = A_Circle_Arm("192.168.1.182")
    # my_arm.move_a_point(8)
    my_arm.run()