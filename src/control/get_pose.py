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

arm = XArmAPI("192.168.1.182")
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

ret = arm.get_position()
print(ret)
ret = arm.get_servo_angle(servo_id=6)
print(f"gripper_position{ret}")

arm.disconnect()