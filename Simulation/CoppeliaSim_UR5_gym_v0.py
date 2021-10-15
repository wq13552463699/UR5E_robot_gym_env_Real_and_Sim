#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 22:33:12 2021

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


# TODO: this environment doesn't include the render part, I still need to fix the delay of coppeliasim
# if you want to render your work, please use render.py
# TODO: add a config.josn file to store all configuration of the environment.
# TODO: write v0 and v1 together to make the envrionment be compatiable with HER and PPO and other algorithms
# together.
DELTA = pi/180.0
REACH_THRESHOLD = 0.08
mu, sigma = 0, 1
MUL_RATE = 2
JOINT_THRESHOLD = 5e-03
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
class CoppeliaSim_UR5_gym_v0(CoppeliaSim_UR5_base,gym.Env):

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
            np.zeros((64,128,3),dtype=np.float32) 
        )
        
        self.high_state = np.array(
            255 * np.ones((64,128,3),dtype=np.float32)
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
        self.was_done = True
        self.num_step = 0   
        self.reset()
        #Need to be improved
        
    def reset(self):
        self.sim_recover()
        self.delay(2)
        # if self.reset_mode:
        #     if self.was_done:
        #         self.target_pos = self.random_gen_one_target_pos()
        #         self.was_done = False
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
        # self.last_dis = copy(self.current_dis)
        self.num_step = 0 
        self.total_reward = 0
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
        pic1 = self.get_image('Side_camera2')
        pic2 = self.get_image('Side_camera3')
        state = np.concatenate((pic1, pic2),axis=1)
        # state = self.get_image('Side_camera2')
        # self.delay(0.3)
        return state
    
    def step(self,action):
        self.apply_action(action)
        
        if not self.stuck_flag:
            self.tip_pos = self.get_tip_position()
            self.current_dis = self.goal_distance(self.tip_pos, self.target_pos)
            self.current_joint_pos = self.get_all_joints_position()
        
            
            self.num_step += 1
            
            ##########################Reward 1####################################
            # reward = (self.current_dis - sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)) / self.current_dis
            ######################################################################
            
            
            ##########################Reward 2####################################
            reward = (self.last_dis - self.current_dis) * 100
            self.total_reward += reward
            # reward =  -self.current_dis
            ######################################################################
            
            ###########################Reward 3###################################
            # if self.current_dis < self.last_dis:
            #     reward = 0
            # elif self.current_dis > self.last_dis:
            #     reward = -1
            ######################################################################
            
            # self.last_dis = copy(self.current_dis)
            next_state = self.get_state()
            done, step_end_status = self.check_done()
            
            ##########################Reward 4####################################
            # if done == 1:
            #     reward = 200.0
            # elif done == 2:
            #     reward = -30.0
            # elif done == 0:
            #     reward = -1.0
            ##########################Reward5 ####################################
            # if done == 1:
            #     reward = 0
            # elif done == 2:
            #     reward = -50
            # elif done ==0:
            #     reward = -1
            #####################################################################    
            
            ##########################Reward2 additional ########################
            if step_end_status == 1:
                reward += (self.total_reward / self.num_step) * 10
            elif step_end_status == 2:
                reward += -30
                
            ##########################Reward 6###################################
            
        
        elif self.stuck_flag:
            next_state = self.get_state()
            reward = -30
            done = True
            step_end_status = 2
            self.stuck_flag = False
            print("Mayday!, I am stuck here")
            
        return next_state, reward, done, {'step_end_status':step_end_status}
        
    
    def apply_action(self, action):
        _joint1_move = action[0] * MOVE_UNIT
        _joint2_move = action[1] * MOVE_UNIT
        _joint3_move = action[2] * MOVE_UNIT
        _joint4_move = action[3] * MOVE_UNIT
        # _joint5_move = action[0][4] * MOVE_UNIT
        self.current_joint_pos[0] += _joint1_move
        self.current_joint_pos[1] += _joint2_move
        self.current_joint_pos[2] += _joint3_move
        self.current_joint_pos[3] += _joint4_move
        # self.current_joint_pos[4] + _joint5_move,
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
    
    def check_done(self):
        ax,ay,az = self.tip_pos[0],self.tip_pos[1],self.tip_pos[2]
        
        if self.current_dis <= REACH_THRESHOLD:
            print("I am reaching the target!!!!!!!!!!!!!!")
            self.was_done = True
            return True, 1
        
        # elif self.num_step >= MAX_STEP:
        #     print('Max time step reached')
        #     return True, 2
        
        elif ax <= ur5_safebox_low[0]:
            print("Touching b3")
            return True, 2
        #In theory, b3 will not be touched anyway
        elif ax >= ur5_safebox_high[0]:
            print("Touching b4")
            return True, 2
        elif ay <= ur5_safebox_low[1]:
            print("Touching b2")
            return True, 2
        elif ay >= ur5_safebox_high[1]:
            print("Touching b1")
            return True, 2
        elif az <= ur5_safebox_low[2]:
            print("Touching table surface")
            return True, 2
        elif az >= ur5_safebox_high[2]:
            print("Touching sky")
            return True, 2
        # In theory, sky will never be touched..... :), it is too high
        
#TODO: Is there any nicer way to do it?
        elif self.current_joint_pos[0] <= joint1[0]:
            print("Joint 1 is reaching the low joint limit")
            return True, 2
        elif self.current_joint_pos[0] >= joint1[1]:
            print("Joint 1 is reaching the high joint limit")
            return True, 2
        
        elif self.current_joint_pos[1] <= joint2[0]:
            print("Joint 2 is reaching the low joint limit")
            return True, 2
        elif self.current_joint_pos[1] >= joint2[1]:
            print("Joint 2 is reaching the high joint limit")
            return True, 2
        #For the real robot, the joint limit for the second joint is [-pi,pi], but in ours application
        #we table, so our joint 2 cannot reach more than pi/2
        
        elif self.current_joint_pos[2] <= joint3[0]:
            print("Joint 3 is reaching the low joint limit")
            return True, 2
        elif self.current_joint_pos[2] >= joint3[1]:
            print("Joint 3 is reaching the high joint limit")
            return True, 2
        
        elif self.current_joint_pos[3] <= joint4[0]:
            print("Joint 4 is reaching the low joint limit")
            return True, 2
        elif self.current_joint_pos[3] >= joint4[1]:
            print("Joint 4 is reaching the high joint limit")
            return True, 2
        
        # elif self.current_joint_pos[4] <= joint5[0]:
        #     print("Joint 5 is reaching the low joint limit")
        #     return 2
        # elif self.current_joint_pos[4] >= joint5[1]:
        #     print("Joint 5 is reaching the high joint limit")
        #     return 2
        
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
    
    # def compute_reward(self, achieved_goal, desired_goal, info):
    #     ax,ay,az = self.tip_pos[0],self.tip_pos[1],self.tip_pos[2]
    #     tx,ty,tz = self.target_pos[0], self.target_pos[1], self.target_pos[2]
    #     self.current_dis = sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)
    #     done = info['done']
    #     step_end_status = info['step_end_status']
        
    #     if self.reward_function == 'CM':
    #         reward = (self.last_dis - self.current_dis) * 100
            
    #     elif self.reward_function == 'DB':
    #         pass
        
    #     elif self.reward_function == 'SR':
    #         if step_end_status == 1:
    #             reward = 0
    #         elif step_end_status == 2:
    #             reward = -1
    #         elif step_end_status ==0:
    #             reward = -1
        
    #     elif self.reward_function == 'FL':
    #         if self.current_dis < self.last_dis:
    #             reward = 0
    #         elif self.current_dis > self.last_dis:
    #             reward = -1
                
    #     if self.award_and_punish:
    #         if step_end_status == 1:
    #             reward += award_value
    def goal_distance(self,goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)