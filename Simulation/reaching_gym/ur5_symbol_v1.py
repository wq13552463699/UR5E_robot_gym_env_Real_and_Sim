# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 16:49:33 2021

@author: 14488
"""

import gym
from gym import spaces
from ur5_kinematic import UR5_kinematic
import numpy as np
from math import pi,sqrt
from copy import copy
from gym.utils import seeding

# TODO: this environment doesn't include the render part, I still need to fix the delay of coppeliasim
# if you want to render your work, please use render.py

DELTA = pi/180.0
REACH_THRESHOLD = 0.05
mu, sigma = 0, 1
MUL_RATE = 2

# You need to modify the following area yourself you specify your robot's condition:
######################Modify following unit yourself##########################
UR5_global_position = [0.4,-0.425,0.65] #UR5's base coordinate in your coordination system
ur5_safebox_high = [0.72,0.275,1.8] # UR5's safety working range's low limit
ur5_safebox_low = [-0.72,-1.025,0.66] # UR5's safety working range's high limit
target_range_low = [-0.48,-0.795,0.663] # Target position's low limit
target_range_high = [-0.02,-0.055,0.663] # Target position's high limit
MOVE_UNIT = 1 * DELTA # Joints' moving unit

# Robot joint's moveing limit:
# TODO: These moving limit still need to be re-colibarated
joint1 = [-pi , pi]
joint2 = [-pi/2, pi/2]
joint3 = [-pi, pi]
joint4 = [-pi, pi]
# joint5 = [-pi, pi] # We are only using 4 joints now.
##############################################################################

action_low = [-10,-10,-10,-10]
action_high = [10,10,10,10]
# TODO: need to have another robot's body checking system, in order that robot's 
# body won't touch safety boundary and self-collision. As to the current one, it 
# can only gurrantee that tip won't touch the boundary.

award_value = 200.0
punish_value = -30.0
# random_action : Gussian, Uniform
# reward function : CM,DB,SR,FL
#CM:closer get more, further lose
#DB: disrance based
# SR: sparse reward
# further lose
last_dis_buffer = 0.0

class ur5_symbol_v1(gym.GoalEnv):
    def __init__(self,
                 reset_to_pre_if_not_reach = True,
                 random_action = 'Gussian',
                 init_joint_pos = np.array([0.0,0.0,0.0,0.0,pi/2,0.0]),
                 reward_type = 'continous',
                 ):
        
        self.kin = UR5_kinematic(UR5_global_position)
        self.reset_mode = reset_to_pre_if_not_reach
        self.random_action = random_action
        self.reward_type = reward_type
        self.seed()
        self.init_joints_pos = init_joint_pos
        self.reset()

        self.low_action = np.array(
            action_low, dtype=np.float32
        )
        
        self.high_action = np.array(
            action_high, dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=self.low_action,
            high=self.high_action,
            dtype=np.float32
        )
        
        obs = self.get_state()
        self.observation_space = spaces.Dict(dict(
        desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
        achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
        observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))
        

    def reset(self):
        self.current_joint_pos = copy(self.init_joints_pos)
        self.target_pos = self.random_gen_one_target_pos()
        return self.get_state()
    
    def get_tip_pos(self):
        coor = self.kin.Forward_ur5(self.current_joint_pos)
        return np.array([coor[0],coor[1],coor[2]])
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def get_state(self):
        tip_pos = self.get_tip_pos()
        state = np.array(
                        [self.current_joint_pos[0], self.current_joint_pos[1],
                         self.current_joint_pos[2], self.current_joint_pos[3],
                         tip_pos[0],tip_pos[1],tip_pos[2],self.target_pos[0],
                         self.target_pos[1],self.target_pos[2]], dtype=np.float32
                        )
        return {
        'observation': state.copy(),
        'achieved_goal': tip_pos.copy(),
        'desired_goal': self.target_pos.copy(),
        }
    
    def random_gen_one_target_pos(self):
        pos = np.random.uniform(target_range_low,target_range_high)
        return pos
    
    def step(self,action):
        self.apply_action(action)
        next_state = self.get_state()
        
        done, step_end_status = self.check_done()
        
        if step_end_status == 1:
            is_success = True
        else:
            is_success = False
        
        info = {
            'step_end_status':step_end_status,
            'is_success':is_success
            }
        
        reward = self.compute_reward(next_state['achieved_goal'],next_state['desired_goal'],info)
        
        return next_state, reward, done, info
    
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
    
    
    def check_done(self):
        tip_pos = self.get_tip_pos()
        ax,ay,az = tip_pos[0],tip_pos[1],tip_pos[2]
        tx,ty,tz = self.target_pos[0], self.target_pos[1], self.target_pos[2]
# TODO:this place need to be re-modified, to match the same format as 'compute_reward'
        if sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2) <= REACH_THRESHOLD:
            print("I am reaching the target!!!!!!!!!!!!!!")
            return True, 1
        
        elif ax <= ur5_safebox_low[0]:
            # print("Touching b3")
            return True, 2
        #In theory, b3 will not be touched anyway
        elif ax >= ur5_safebox_high[0]:
            # print("Touching b4")
            return True, 2
        elif ay <= ur5_safebox_low[1]:
            # print("Touching b2")
            return True, 2
        elif ay >= ur5_safebox_high[1]:
            # print("Touching b1")
            return True, 2
        elif az <= ur5_safebox_low[2]:
            # print("Touching table surface")
            return True, 2
        elif az >= ur5_safebox_high[2]:
            # print("Touching sky")
            return True, 2
        # In theory, sky will never be touched..... :), it is too high
        
#TODO: Is there any nicer way to do it?
        elif self.current_joint_pos[0] <= joint1[0]:
            # print("Joint 1 is reaching the low joint limit")
            return True, 2
        elif self.current_joint_pos[0] >= joint1[1]:
            # print("Joint 1 is reaching the high joint limit")
            return True, 2
        
        elif self.current_joint_pos[1] <= joint2[0]:
            # print("Joint 2 is reaching the low joint limit")
            return True, 2
        elif self.current_joint_pos[1] >= joint2[1]:
            # print("Joint 2 is reaching the high joint limit")
            return True, 2
        #For the real robot, the joint limit for the second joint is [-pi,pi], but in ours application
        #we table, so our joint 2 cannot reach more than pi/2
        
        elif self.current_joint_pos[2] <= joint3[0]:
            # print("Joint 3 is reaching the low joint limit")
            return True, 2
        elif self.current_joint_pos[2] >= joint3[1]:
            # print("Joint 3 is reaching the high joint limit")
            return True, 2
        
        elif self.current_joint_pos[3] <= joint4[0]:
            # print("Joint 4 is reaching the low joint limit")
            return True, 2
        elif self.current_joint_pos[3] >= joint4[1]:
            # print("Joint 4 is reaching the high joint limit")
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
            action = np.array([MUL_RATE * np.random.normal(mu, sigma),
                      MUL_RATE * np.random.normal(mu, sigma),
                      MUL_RATE * np.random.normal(mu, sigma),
                      MUL_RATE * np.random.normal(mu, sigma)])
            
        if self.random_action == "Uniform":
            action = np.random.uniform(action_low,action_high)
            
        return action
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        d = self.goal_distance(achieved_goal, desired_goal)
        if self.reward_type == 'sparse':
            return -(d > REACH_THRESHOLD).astype(np.float32)
        else:
            return -d
    
    def goal_distance(self,goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)
            
            
            