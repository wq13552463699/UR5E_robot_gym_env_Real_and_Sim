#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 22:18:13 2021

@author: qiang
"""
from ur5_kinematic import UR5_kinematic
import numpy as np
from math import pi,sqrt
from copy import copy


# TODO: this environment doesn't include the render part, I still need to fix the delay of coppeliasim
# if you want to render your work, please use render.py

UR5_global_position = [0.4,-0.425,0.65]
ur5_safebox_high = [0.72,0.275,1.8]
ur5_safebox_low = [-0.72,-1.025,0.66]
target_range_low = [-0.48,-0.795,0.663]
target_range_high = [-0.02,-0.055,0.663]
target_range_x = [-0.48,-0.434,-0.388,-0.342,-0.296,-0.25,-0.204,-0.158,-0.112,-0.066]
target_range_y = [-0.721,-0.647,-0.573,-0.499,-0.425,-0.351,-0.277,-0.203,-0.129,-0.055]
DELTA = pi/180.0
REACH_THRESHOLD = 0.05

MOVE_UNIT = 1 * DELTA

joint1 = [-pi , pi]
joint2 = [-pi/2, pi/2]
joint3 = [-pi, pi]
joint4 = [-pi, pi]
joint5 = [-pi, pi]

STATE_DIM = 4 + 3 + 3


class symbol_env_continous():
    def __init__(self):
        self.init_joints_pos = [0.0,0.0,0.0,0.0,pi/2,0.0]
        self.state_dim = STATE_DIM
        self.current_joint_pos = copy(self.init_joints_pos)
        self.kinema = UR5_kinematic(UR5_global_position)
        self.reset()

    # def get_state_dim(self):
    #     return self.state_dim

    # def get_action_dim(self):
    #     return 12

    def get_tip_pos(self):
        pos = self.kinema.Forward_ur5(self.current_joint_pos)
        return [pos[0],pos[1],pos[2]]

    def reset(self):
        self.current_joint_pos = copy(self.init_joints_pos)
        self.target_pos = self.random_gen_one_target_pos()
        tip_pos = self.get_tip_pos()
        ax, ay, az = tip_pos[0], tip_pos[1], tip_pos[2]
        tx, ty, tz = self.target_pos[0], self.target_pos[1], self.target_pos[2]
        self.current_dis = sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)
        
        self.last_dis = copy(self.current_dis)
        
        return self.get_state()

    def reset_o(self):
        # This function will reset the arm but don't ramdomly reset the position of the target, as
        # long as the arm can get the target, the target will be reset.
        self.current_joint_pos = copy(self.init_joints_pos)
        # print(self.init_joints_pos)
        tip_pos = self.get_tip_pos()
        ax, ay, az = tip_pos[0], tip_pos[1], tip_pos[2]
        tx, ty, tz = self.target_pos[0], self.target_pos[1], self.target_pos[2]
        
        self.current_dis = sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)
        
        self.last_dis = copy(self.current_dis)
        
        return self.get_state()

    def random_gen_one_target_pos(self):
        pos = np.random.uniform(target_range_low,target_range_high)
        self.target_pos = pos.tolist()
        return self.target_pos
    
    def randon_get_sparse_pos(self):
        num_x = np.random.randint(10)
        num_y = np.random.randint(10)
        # print(num_x,num_y)
        self.target_pos = [target_range_x[num_x],target_range_y[num_y],0.663]
        return self.target_pos

    def get_state(self):
        data = np.concatenate([[self.current_joint_pos[0],self.current_joint_pos[1],
                                self.current_joint_pos[2],self.current_joint_pos[3]]
                               ,self.get_tip_pos(),self.target_pos])
        data.reshape(len(data),-1)
        # data = data.tolist()
        return data

    def step(self,action):
        self.current_joint_pos = self.action_space(action)
        tip_pos = self.get_tip_pos()
        ax,ay,az = tip_pos[0],tip_pos[1],tip_pos[2]
        tx,ty,tz = self.target_pos[0], self.target_pos[1], self.target_pos[2]
        self.current_dis = sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)
        
        ##########################Reward 1####################################
        # reward = (self.current_dis - sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)) / self.current_dis
        ######################################################################
        
        
        ##########################Reward 2####################################
        reward = (self.last_dis - self.current_dis) * 100
        ######################################################################
        
        ###########################Reward 3###################################
        # if self.current_dis < self.last_dis:
        #     reward = 0
        # elif self.current_dis > self.last_dis:
        #     reward = -1
        ######################################################################
        
        self.last_dis = copy(self.current_dis)
        next_state = self.get_state()
        done = self.check_done()
        
        
        
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
        if done == 1:
            reward += 200.0
        elif done == 2:
            reward += -30.0
        
        return next_state, reward, done

    def check_done(self):
        tip_pos = self.get_tip_pos()
        ax,ay,az = tip_pos[0],tip_pos[1],tip_pos[2]
        tx,ty,tz = self.target_pos[0], self.target_pos[1], self.target_pos[2]
        
        if sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2) <= REACH_THRESHOLD:
            print("I am reaching the target!!!!!!!!!!!!!!")
            return 1
        
        
        elif ax <= ur5_safebox_low[0]:
            print("Touching b3")
            return 2
        #In theory, b3 will not be touched anyway
        elif ax >= ur5_safebox_high[0]:
            print("Touching b4")
            return 2
        elif ay <= ur5_safebox_low[1]:
            print("Touching b2")
            return 2
        elif ay >= ur5_safebox_high[1]:
            print("Touching b1")
            return 2
        elif az <= ur5_safebox_low[2]:
            print("Touching table surface")
            return 2
        elif az >= ur5_safebox_high[2]:
            print("Touching sky")
            return 2
        # In theory, sky will never be touched..... :), it is too high
        
#TODO: Is there any nicer way to do it?
        elif self.current_joint_pos[0] <= joint1[0]:
            print("Joint 1 is reaching the low joint limit")
            return 2
        elif self.current_joint_pos[0] >= joint1[1]:
            print("Joint 1 is reaching the high joint limit")
            return 2
        
        elif self.current_joint_pos[1] <= joint2[0]:
            print("Joint 2 is reaching the low joint limit")
            return 2
        elif self.current_joint_pos[1] >= joint2[1]:
            print("Joint 2 is reaching the high joint limit")
            return 2
        #For the real robot, the joint limit for the second joint is [-pi,pi], but in ours application
        #we table, so our joint 2 cannot reach more than pi/2
        
        elif self.current_joint_pos[2] <= joint3[0]:
            print("Joint 3 is reaching the low joint limit")
            return 2
        elif self.current_joint_pos[2] >= joint3[1]:
            print("Joint 3 is reaching the high joint limit")
            return 2
        
        elif self.current_joint_pos[3] <= joint4[0]:
            print("Joint 4 is reaching the low joint limit")
            return 2
        elif self.current_joint_pos[3] >= joint4[1]:
            print("Joint 4 is reaching the high joint limit")
            return 2
        
        # elif self.current_joint_pos[4] <= joint5[0]:
        #     print("Joint 5 is reaching the low joint limit")
        #     return 2
        # elif self.current_joint_pos[4] >= joint5[1]:
        #     print("Joint 5 is reaching the high joint limit")
        #     return 2

        else:
            return 0

    def action_space(self, action):
        
        _joint1_move = action[0][0] * MOVE_UNIT
        _joint2_move = action[0][1] * MOVE_UNIT
        _joint3_move = action[0][2] * MOVE_UNIT
        _joint4_move = action[0][3] * MOVE_UNIT
        # _joint5_move = action[0][4] * MOVE_UNIT
        
        New_pos = [self.current_joint_pos[0] + _joint1_move,
                   self.current_joint_pos[1] + _joint2_move,
                   self.current_joint_pos[2] + _joint3_move,
                   self.current_joint_pos[3] + _joint4_move,
                   # self.current_joint_pos[4] + _joint5_move,
                   pi/2,
                   0
                   ]
        
        return New_pos
        
        
        