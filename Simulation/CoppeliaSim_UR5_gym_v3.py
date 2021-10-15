#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 01:13:42 2021

@author: qiang
"""

from CoppeliaSim_UR5_base import CoppeliaSim_UR5_base
import gym
from gym import spaces
from ur5_kinematic import UR5_kinematic
import numpy as np
from math import pi,sqrt
from copy import copy
from gym.utils import seeding
import time
# TODO: add a config.josn file to store all configuration of the environment.
# TODO: write v0 and v1 together to make the envrionment be compatiable with HER and PPO and other algorithms
# together.
DELTA = pi/180.0
REACH_THRESHOLD = 0.08
mu, sigma = 0, 1
MUL_RATE = 2
JOINT_THRESHOLD = 5e-03
STAY_THRESHOLD = 0.1
STAY_FLAG = 30
# You need to modify the following area yourself you specify your robot's condition:
######################Modify following unit yourself##########################
UR5_global_position = [0.4,-0.425,0.65] #UR5's base coordinate in your coordination system
ur5_safebox_high = [0.72,0.275,1.8] # UR5's safety working range's low limit
ur5_safebox_low = [-0.72,-1.025,0.66] # UR5's safety working range's high limit
MOVE_UNIT = 2 * DELTA # Joints' moving unit
TIMEOUT = 2
MAX_STEP = 200
# Robot joint's moveing limit:
# TODO: These moving limit still need to be re-colibarated
joint1 = [-pi , pi]
joint2 = [-pi/2, pi/2]
joint3 = [-pi, pi]
joint4 = [-pi, pi]
# joint5 = [-pi, pi] # We are only using 4 joints now.
##############################################################################

action_low = [-5,-5,-5,-5]
action_high = [5,5,5,5]
# TODO: need to have another robot's body checking system, in order that robot's 
# body won't touch safety boundary and self-collision. As to the current one, it 
# can only gurrantee that tip won't touch the boundary.

# random_action : Gussian, Uniform

# TODO : there should be a way to improve the taking image.
class CoppeliaSim_UR5_gym_v3(CoppeliaSim_UR5_base,gym.Env):

    def __init__(self,
                 reset_to_pre_if_not_reach = True,
                 random_action = 'Gussian',
                 scene_path = None,
                 scene_side = None):
        
        super().__init__(scene_path=scene_path,scene_side=scene_side)        

        self.reset_mode = reset_to_pre_if_not_reach
        self.random_action = random_action
        self.seed()
        self.sim_start()
        self.stuck_flag = False
        
        self.low_action = np.array(
            action_low, dtype=np.float32
        )
        
        self.high_action = np.array(
            action_high, dtype=np.float32
        )
        
        self.low_state = np.array(
            np.zeros((128,128,3),dtype=np.float32) 
        )
        
        self.high_state = np.array(
            255 * np.ones((128,128,3),dtype=np.float32)
        )
        
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )
        
        
        self.action_space = spaces.Box(
            low=self.low_action,
            high=self.high_action,
            dtype=np.float32
        )
        
        self.init_joints_pos = np.array([0.0,0.0,0.0,0.0,pi/2,0.0])
        
        #Reset
        self.reset()
        #Need to be improved
        
    def reset(self):
        self.sim_recover()
        self.delay(2)
        self.set_target_to_random_pos()
        self.delay(0.3)
        while True:
            self.movej(self.init_joints_pos)
            if self.check_move(self.init_joints_pos):
                break
        self.current_joint_pos = copy(self.init_joints_pos)
        self.tip_pos = self.get_tip_position()
        self.target_pos = self.get_target_position()
        self.current_dis = self.goal_distance(self.tip_pos, self.target_pos)
        self.num_step = 0 
        self.tra_num_step = -1
        self.last_dis = copy(self.current_dis)
        self.tra_total_reward = 0
        self.stay_loc = []
        self.sf = 0
        return self.get_state()

    def reset_pos(self):
        self.sim_recover()
        self.delay(2)
        self.set_target_to_random_pos()
        self.delay(0.3)
        while True:
            self.movej(self.init_joints_pos)
            if self.check_move(self.init_joints_pos):
                break
        self.current_joint_pos = copy(self.init_joints_pos)
        self.tip_pos = self.get_tip_position()
        self.target_pos = self.get_target_position()
        self.current_dis = self.goal_distance(self.tip_pos, self.target_pos)
        self.last_dis = copy(self.current_dis)
        self.tra_num_step = -1
        self.stay_loc = []
        self.tra_total_reward = 0
        self.sf = 0
        return self.get_state()

    def delay(self,x):
        time.sleep(x)
        
    def check_move(self,plc):
        actual = self.get_all_joints_position()
        if self.goal_distance(plc, actual) <= JOINT_THRESHOLD:
            return True
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def get_state(self):
        pic1 = self.get_image('Side_camera1')
        # pic2 = self.get_image('Side_camera2')
        # state = np.concatenate((pic1, pic2),axis=1)
        return pic1
    
    def step(self,action):
        self.apply_action(action)
        
        
        if not self.stuck_flag:
            done, step_end_status = self.check_done()
            
            if step_end_status == 2:
                reward = -150
                next_state = self.get_state()
                self.reset_pos()
                
            elif step_end_status == 3:
                self.current_dis = self.goal_distance(self.tip_pos, self.target_pos)
                reward = (self.last_dis - self.current_dis) * 100
                next_state = self.get_state()
                self.last_dis = copy(self.current_dis)
                self.reset_pos()
                
            else:
                
                self.current_dis = self.goal_distance(self.tip_pos, self.target_pos)
                # self.current_joint_pos = self.get_all_joints_position()
                next_state = self.get_state()
                reward = (self.last_dis - self.current_dis) * 100
                
                # print(self.last_dis - self.current_dis)
                
                self.last_dis = copy(self.current_dis)
                self.tra_total_reward += reward
                if step_end_status == 1:
                    reward += (2000 / self.tra_num_step)
                    self.reset_pos()
        
        elif self.stuck_flag:
            next_state = self.get_state()
            done = False
            step_end_status = 2
            reward = -50
            self.stuck_flag = False
            print("\nMayday!, I am stuck here")
            self.reset_pos()
             
        return next_state, reward, done, {'step_end_status':step_end_status}
        
    
    def apply_action(self, action):
        _joint1_move = action[0] * MOVE_UNIT
        _joint2_move = action[1] * MOVE_UNIT
        _joint3_move = action[2] * MOVE_UNIT
        _joint4_move = action[3] * MOVE_UNIT
        self.current_joint_pos[0] += _joint1_move
        self.current_joint_pos[1] += _joint2_move
        self.current_joint_pos[2] += _joint3_move
        self.current_joint_pos[3] += _joint4_move
        t1 = time.time()
        while True:
            t2 = time.time()
            if (t2 - t1) > TIMEOUT:
                self.stuck_flag = True
                break
            else:
                self.movej(self.current_joint_pos)
                if self.check_move(self.current_joint_pos):
                    break
        self.tip_pos = self.get_tip_position()
        self.stay_loc.append(self.tip_pos)
        self.tra_num_step += 1
        # print(self.tra_num_step)
        # print(self.stay_loc)
    def check_done(self):
        ax,ay,az = self.tip_pos[0],self.tip_pos[1],self.tip_pos[2]
        
        if self.current_dis <= REACH_THRESHOLD:
            print("\nI am reaching the target!!!!!!!!!!!!!!")
            return False, 1
        
        elif self.num_step >= MAX_STEP:
            print('Max time step reached')
            return True, 2
        
        
        
        elif ax <= ur5_safebox_low[0]:
            print("\nTouching b3")
            return False, 2
        #In theory, b3 will not be touched anyway
        elif ax >= ur5_safebox_high[0]:
            print("\nTouching b4")
            return False, 2
        elif ay <= ur5_safebox_low[1]:
            print("\nTouching b2")
            return False, 2
        elif ay >= ur5_safebox_high[1]:
            print("\nTouching b1")
            return False, 2
        elif az <= ur5_safebox_low[2]:
            print("\nTouching table surface")
            return False, 2
        elif az >= ur5_safebox_high[2]:
            print("\nTouching sky")
            return False, 2
        # In theory, sky will never be touched..... :), it is too high
        
#TODO: Is there any nicer way to do it?
        elif self.current_joint_pos[0] <= joint1[0]:
            print("\nJoint 1 is reaching the low joint limit")
            return False, 2
        elif self.current_joint_pos[0] >= joint1[1]:
            print("\nJoint 1 is reaching the high joint limit")
            return False, 2
        
        elif self.current_joint_pos[1] <= joint2[0]:
            print("\nJoint 2 is reaching the low joint limit")
            return False, 2
        elif self.current_joint_pos[1] >= joint2[1]:
            print("\nJoint 2 is reaching the high joint limit")
            return False, 2
        #For the real robot, the joint limit for the second joint is [-pi,pi], but in ours application
        #we have table, so our joint 2 cannot reach more than pi/2
        
        elif self.current_joint_pos[2] <= joint3[0]:
            print("\nJoint 3 is reaching the low joint limit")
            return False, 2
        elif self.current_joint_pos[2] >= joint3[1]:
            print("\nJoint 3 is reaching the high joint limit")
            return False, 2
        
        elif self.current_joint_pos[3] <= joint4[0]:
            print("\nJoint 4 is reaching the low joint limit")
            return False, 2
        elif self.current_joint_pos[3] >= joint4[1]:
            print("\nJoint 4 is reaching the high joint limit")
            return False, 2
                
        elif self.tra_num_step >= STAY_FLAG:
            return self.check_stay()
        
        else:
            return False, 0
    
    def random_action(self):
        if self.random_action == "Gussian":
            action = [MUL_RATE * np.random.normal(mu, sigma),
                      MUL_RATE * np.random.normal(mu, sigma),
                      MUL_RATE * np.random.normal(mu, sigma),
                      MUL_RATE * np.random.normal(mu, sigma)]
            
        if self.random_action == "Uniform":
            action = np.random.uniform(action_low,action_high)
            
        return action
    
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