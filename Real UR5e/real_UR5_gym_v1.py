# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 16:51:15 2021

@author: 14488
"""

#需要调试：
# 1. ur5_safebox
# 2. UR5_global_position
# 3. Init joint pos


import gym
from gym import spaces
import numpy as np
from math import pi,sqrt
from copy import copy
from gym.utils import seeding
import urx
from IntelRealSense import IntelRealSense
from cnn_utils import *
import cv2
import cnn_utils
from Forward_kin_v1 import UR5e_kin

# TODO: this environment doesn't include the render part, I still need to fix the delay of coppeliasim
# if you want to render your work, please use render.py
# TODO: add a config.josn file to store all configuration of the environment.
# TODO: write v0 and v1 together to make the envrionment be compatiable with HER and PPO and other algorithms
# together.
STAY_FLAG = 30
STAY_THRESHOLD = 0.1
DELTA = pi/180.0
REACH_THRESHOLD = 0.05
mu, sigma = 0, 1
MUL_RATE = 2
JOINT_THRESHOLD = 5e-03
label_coordinate = [[25.883026123046875, 25.764095306396484], 
                    [253.16226196289062, 18.941144943237305], 
                    [24.80422019958496, 157.13990783691406], 
                    [249.27520751953125, 156.4306640625]]
global_init = [-pi/2,-pi/2,0,-pi/2,0,0]
boundary_buffer = 0.05
target_height = 0.06
# You need to modify the following area yourself you specify your robot's condition:
######################Modify following unit yourself##########################
UR5_global_position = [0,0,0] #UR5's base coordinate in your coordination system
ur5_boundary_low = [-0.4,-0.915,0.005]
ur5_boundary_high = [0.4,0.425,0.87]

ur5_safebox_low = [ur5_boundary_low[0]+boundary_buffer,ur5_boundary_low[1]+boundary_buffer,ur5_boundary_low[2]+boundary_buffer] # UR5's safety working range's low limit
ur5_safebox_high =  [ur5_boundary_high[0]-boundary_buffer,ur5_boundary_high[1]-boundary_buffer,ur5_boundary_high[2]-boundary_buffer]# UR5's safety working range's high limit

MOVE_UNIT = 1.5 * DELTA # Joints' moving unit
TIMEOUT = 2
MAX_STEP = 300


# Robot joint's moveing limit:
# TODO: These moving limit still need to be re-colibarated
joint1_limit = [-pi, pi]
joint2_limit = [-0.4*pi,0.4*pi]
joint3_limit = [-0.75*pi,0.75*pi]
joint4_limit = [-pi,pi]

joint1 = [joint1_limit[0]  +global_init[0] , joint1_limit[1]  +global_init[0]]
joint2 = [joint2_limit[0]  +global_init[1],  joint2_limit[1]  +global_init[1]]
joint3 = [joint3_limit[0]  +global_init[2],  joint3_limit[1]  +global_init[2]]
joint4 = [joint4_limit[0]  +global_init[3],  joint4_limit[1]  +global_init[3]]
# joint5 = [-pi, pi] # We are only using 4 joints now.
##############################################################################

action_low = [-4,-4,-4,-4]
action_high = [4,4,4,4]
# TODO: need to have another robot's body checking system, in order that robot's 
# body won't touch safety boundary and self-collision. As to the current one, it 
# can only gurrantee that tip won't touch the boundary.

# random_action : Gussian, Uniform

# TODO : there should be a way to improve the taking image.
class real_UR5_gym_v1(gym.Env):

    def __init__(self,
                 reset_to_pre_if_not_reach = True,
                 random_action = 'Gussian',
                 observation_type = 'joint', # joint/image/imageANDjoint
                 test_env = False,
                 check_target = False,
                 ):
        
        # super().__init__(scene_path=scene_path,scene_side=scene_side)        
        self.kin = UR5e_kin(48)
        self.seed()
        
        self.metadata = cnn_utils.get_metadata()
        self.predictor = cnn_utils.get_predictor()
        self.observation_type = observation_type
        self.test_env = test_env
        self.check_target = check_target
        
        self.rob = urx.Robot("192.168.75.128", use_rt=True)
        self.rob.set_tcp((0, 0, 0.02, 0, 0, 0))
        self.rob.set_payload(2, (0, 0, 0.1))
        print('UR5 robot has been connected')
        
        self.low_action = np.array(
            action_low, dtype=np.float32
        )
        
        self.high_action = np.array(
            action_high, dtype=np.float32
        )
        
        if self.observation_type == 'image':
            self.low_state = np.array(
                np.zeros((270,175,3),dtype=np.float32) 
            )
            
            self.high_state = np.array(
                255 * np.ones((270,175,3),dtype=np.float32)
            )
            
            self.observation_space = spaces.Box(
                low=self.low_state,
                high=self.high_state,
                dtype=np.float32
            )
        elif self.observation_type == 'joint':
            self.low_state = np.array(
            [joint1[0], joint2[0],joint3[0],joint4[0],-np.inf,-np.inf,-np.inf,-np.inf,
             -np.inf,-np.inf], dtype=np.float32
            )
        
            self.high_state = np.array(
                [joint1[1], joint2[1],joint3[1],joint4[1],np.inf,np.inf,np.inf,np.inf,
                 np.inf,np.inf],dtype=np.float32 # TODO: This place still need to be improved, the np.inf's place should be the target's positions.
            )
        
            self.observation_space = spaces.Box(
                low=self.low_state,
                high=self.high_state,
                dtype=np.float32
            )
        
        self.IRS = IntelRealSense()
        
        self.action_space = spaces.Box(
            low=self.low_action,
            high=self.high_action,
            dtype=np.float32
        )
        
        self.init = [0,-0.35,-1.5,0.5,pi/2,0]
        self.outside_camera = [-1.5707233587848108, -1.2869056028178711, -1.4390153884887695, -1.0708845418742676, 1.5706801414489746, 2.2411346435546875e-05]

        
        init_pos = []
        for a in range(6):
            init_pos.append(global_init[a] + self.init[a])
        self.init_joints_pos = np.array(init_pos)
        # self.reset()
        #Need to be improved
        
    def reset(self):
        self.rob.movej(self.outside_camera,acc=0.75, vel=0.5)
        input('Please move the target to a new position, and then press enter')
        self.target_pos = self.get_target_pos()
        self.rob.movej(self.init_joints_pos,acc=0.75, vel=0.5)
        self.current_joint_pos = self.rob.getj(wait=True)
        tp = self.rob.getl(wait=True)
        self.tip_pos = np.array([tp[0],tp[1],tp[2]])
        self.current_dis = self.goal_distance(self.tip_pos, self.target_pos)
        self.last_dis = copy(self.current_dis)
        self.num_step = 0
        self.tra_num_step = 0
        return self.get_state()
    
    def reset_pos(self):
        temp = copy(self.num_step)
        self.reset()
        self.num_step = temp
        return self.get_state()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def get_state(self):
        if self.observation_type == 'image':
            state = self.IRS.get_rbg(crop=[40,215,25,295]) #应该先剪裁然后改变像素
        elif self.observation_type == 'joint':
            tip_pos = self.rob.getl()
            state = np.array(
                            [self.current_joint_pos[0], self.current_joint_pos[1],
                             self.current_joint_pos[2], self.current_joint_pos[3],
                             tip_pos[0],tip_pos[1],tip_pos[2],self.target_pos[0],
                             self.target_pos[1],self.target_pos[2]], dtype=np.float32
                            )
        return state
    
    def step(self,action):
    
        self.apply_action(action)
        
        done, step_end_status = self.check_done()
        
        if step_end_status == 2:
            reward = -50
            next_state = self.get_state()
            self.reset_pos()
                    
        else:
            self.current_dis = self.goal_distance(self.tip_pos, self.target_pos)
            self.current_joint_pos = self.rob.getj(wait=True)
            next_state = self.get_state()
            reward = (self.last_dis - self.current_dis) * 100
            self.last_dis = copy(self.current_dis)
            if step_end_status == 1:
                reward += (2000 / self.tra_num_step)
                self.reset_pos()
    
        return next_state, reward, done, {'step_end_status':step_end_status}
        
    def get_target_pos(self):
        # 这边需要改一下，大体上就是将四个label的位置提前记录下来，然后储存在CSV文件中。
        #使用的时候进行校准和读取。或者直接命名变量。
        # 需要手动得到label的详细的rgb图像的坐标位置和现实世界的坐标位置。
        img = self.IRS.get_rbg(crop=[40,215,25,295])
        img = cv2.flip(img, -1)
        outputs = self.predictor(img)
        coordinates = outputs["instances"].pred_boxes.get_centers().numpy().tolist()
        cub, _ = cnn_utils.get_all_objects_coordinate(coordinates,[],label_coordinate)
        if len(cub) == 0:
            print('Nothing was detected in this episode, please move the target to a easier place as soon as possiable')
            
        return np.array([cub[0][0], cub[0][1],target_height])
    
    def apply_action(self, action):
        _joint1_move = action[0] * MOVE_UNIT
        _joint2_move = action[1] * MOVE_UNIT
        _joint3_move = action[2] * MOVE_UNIT
        _joint4_move = action[3] * MOVE_UNIT
        self.current_joint_pos[0] += _joint1_move
        self.current_joint_pos[1] += _joint2_move
        self.current_joint_pos[2] += _joint3_move
        self.current_joint_pos[3] += _joint4_move
        self.rob.movej(self.current_joint_pos,acc=0.75, vel=0.5)
        self.num_step += 1
        tp = self.rob.getl(wait=True)
        self.tip_pos = np.array([tp[0],tp[1],tp[2]])
        self.tra_num_step += 1

    def check_done(self):
        
        point_pos = self.kin.Get48_global_coor(self.current_joint_pos)
        x = []
        y = []
        z = []
        for i in range(len(point_pos)):
            x.append(point_pos[i][0][0])
            y.append(point_pos[i][0][1])
            z.append(point_pos[i][0][2])
        
        if self.current_dis <= REACH_THRESHOLD:
            print("\nI am reaching the target!!!!!!!!!!!!!!")
            return True, 1
        
        if self.num_step >= MAX_STEP:
          print('Max time step reached')
          return True, 2
        
        elif any(t <= ur5_safebox_low[0] for t in x):
            print("\nTouching x low")
            return False, 2
        elif any(t >= ur5_safebox_high[0] for t in x):
            print("\nTouching x high")
            return False, 2
        elif any(t <= ur5_safebox_low[1] for t in y):
            print("\nTouching y low")
            return False, 2
        elif any(t >= ur5_safebox_high[1] for t in y):
            print("\nTouching y high")
            return False, 2
        elif any(t <= ur5_safebox_low[2] for t in z):
            print("\nTouching z low")
            return False, 2
        elif any(t >= ur5_safebox_high[2] for t in z):
            print("\nTouching z high")
            return False, 2
  
#TODO: Need to figure out a good way to solve self-collision
        # elif self.current_joint_pos[0] <= joint1[0]:
        #     print("\nJoint 1 is reaching the low joint limit")
        #     return False, 2
        # elif self.current_joint_pos[0] >= joint1[1]:
        #     print("\nJoint 1 is reaching the high joint limit")
        #     return False, 2
        
        # elif self.current_joint_pos[1] <= joint2[0]:
        #     print("\nJoint 2 is reaching the low joint limit")
        #     return False, 2
        # elif self.current_joint_pos[1] >= joint2[1]:
        #     print("\nJoint 2 is reaching the high joint limit")
        #     return False, 2
        #For the real robot, the joint limit for the second joint is [-pi,pi], but in ours application
        #we have table, so our joint 2 cannot reach more than pi/2
        
        elif self.current_joint_pos[2] <= joint3[0]:
            print("\nJoint 3 is reaching the low joint limit")
            return False, 2
        elif self.current_joint_pos[2] >= joint3[1]:
            print("\nJoint 3 is reaching the high joint limit")
            return False, 2
        
        # elif self.current_joint_pos[3] <= joint4[0]:
        #     print("\nJoint 4 is reaching the low joint limit")
        #     return False, 2
        # elif self.current_joint_pos[3] >= joint4[1]:
        #     print("\nJoint 4 is reaching the high joint limit")
        #     return False, 2
        
        # elif self.tra_num_step >= STAY_FLAG:
        #     return self.check_stay()
        
        else:
            return False, 0
    
    # def random_action(self):
    #     if self.random_action == "Gussian":
    #         action = [MUL_RATE * np.random.normal(mu, sigma),
    #                   MUL_RATE * np.random.normal(mu, sigma),
    #                   MUL_RATE * np.random.normal(mu, sigma),
    #                   MUL_RATE * np.random.normal(mu, sigma)]
            
    #     if self.random_action == "Uniform":
    #         action = np.random.uniform(action_low,action_high)
            
    #     return action
    
    def goal_distance(self,goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)
    
    def check_stay(self):
        sf = 0
        for i in range(30):
            if self.goal_distance(self.stay_loc[self.tra_num_step],self.stay_loc[self.tra_num_step-i]) <= STAY_THRESHOLD:
                sf += 1
        # print(sf)
        if sf >= 25:
            print("\nThe robot staying there without moving, resetting")
            return False, 3
            print(sf)
        else:
            return False, 0