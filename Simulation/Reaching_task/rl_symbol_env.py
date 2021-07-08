from env_base import UR5_env
from ur5_kinematic import UR5_kinematic
# from kinematic import *
import numpy as np
from math import pi,sqrt
from copy import copy

UR5_global_position = [0.4,-0.425,0.65]
ur5_safebox_high = [0.72,0.275,1.8]
ur5_safebox_low = [-0.72,-1.025,0.66]
target_ranget_low = [-0.48,-0.795,0.663]
target_range_high = [-0.02,-0.055,0.663]
target_range_x = [-0.48,-0.434,-0.388,-0.342,-0.296,-0.25,-0.204,-0.158,-0.112,-0.066]
target_range_y = [-0.721,-0.647,-0.573,-0.499,-0.425,-0.351,-0.277,-0.203,-0.129,-0.055]
DELTA = pi/180.0
REACH_THRESHOLD = 0.1

class symbol_env():
    def __init__(self):
        self.init_joints_pos = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.state_dim = 6 + 3 + 3
        self.current_joint_pos = copy(self.init_joints_pos)
        self.kinema = UR5_kinematic(UR5_global_position)
        self.reset()
        
        self.last_dis = 0

    def get_state_dim(self):
        return self.state_dim

    def get_action_dim(self):
        return 12

    def get_tip_pos(self):
        pos = self.kinema.Forward_ur5(self.current_joint_pos)
        return [pos[0],pos[1],pos[2]]

    def reset(self):
        self.current_joint_pos = copy(self.init_joints_pos)
        # print(self.init_joints_pos)
        self.target_pos = self.randon_get_sparse_pos()
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
        pos = np.random.uniform(target_ranget_low,target_range_high)
        self.target_pos = pos.tolist()
        return self.target_pos
    
    def randon_get_sparse_pos(self):
        num_x = np.random.randint(10)
        num_y = np.random.randint(10)
        # print(num_x,num_y)
        self.target_pos = [target_range_x[num_x],target_range_y[num_y],0.663]
        return self.target_pos

    def get_state(self):
        data = np.concatenate([self.current_joint_pos,self.get_tip_pos(),self.target_pos])
        data.reshape(len(data),-1)
        return data

    def step(self,action):
        move_joint, angle = self.action_space(action)
        self.current_joint_pos[move_joint - 1] += angle
        self.current_joint_pos[move_joint - 1] = self.current_joint_pos[move_joint - 1] % (2*pi)
        tip_pos = self.get_tip_pos()
        ax,ay,az = tip_pos[0],tip_pos[1],tip_pos[2]
        tx,ty,tz = self.target_pos[0], self.target_pos[1], self.target_pos[2]
        self.current_dis = sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)
        
        ##########################Reward 1####################################
        # reward = (self.dis - sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)) / self.dis
        ######################################################################
        
        
        ##########################Reward 2####################################
        # Always use action 3
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
        # if done == 1:
        #     reward = 0
        # elif done == 2:
        #     reward = 0
        # elif done == 0:
        #     reward = -1
        # if done == 1:
        #     reward = 0
        # elif done == 2:
        #     reward = -50
        # elif done ==0:
        #     reward = -1
        # print("c",self.current_dis)
        # print("l",self.last_dis)
        # print(reward)
        
        return next_state, reward, done

    def check_done(self):
        tip_pos = self.get_tip_pos()
        ax,ay,az = tip_pos[0],tip_pos[1],tip_pos[2]
        #print(ax,ay,az)
        tx,ty,tz = self.target_pos[0], self.target_pos[1], self.target_pos[2]
        if sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2) <= REACH_THRESHOLD:
            print("I am reaching the target!!!!!!!!!!!!!!")
            return 1
        elif ax <= ur5_safebox_low[0] or ax >= ur5_safebox_high[0]:
            print("Touching b1")
            return 2
        elif ay <= ur5_safebox_low[1] or ay >= ur5_safebox_high[1]:
            print("Touching b2")
            return 2
        elif az <= ur5_safebox_low[2] or az >= ur5_safebox_high[2]:
            print("Touching b3")
            return 2
        else:
            return 0

    def action_space(self, action):
        #max_pos = np.argmax(act)
        #action = max_pos + 1
        if action == 0:
            return 1, (5 * DELTA)  # joint degree
        if action == 1:
            return 1, (-5 * DELTA)
        if action == 2:
            return 2, (5 * DELTA)
        if action == 3:
            return 2, (-5 * DELTA)
        if action == 4:
            return 3, (5 * DELTA)
        if action == 5:
            return 3, (-5 * DELTA)
        if action == 6:
            return 4, (5 * DELTA)
        if action == 7:
            return 4, (-5 * DELTA)
        if action == 8:
            return 5, (5 * DELTA)
        if action == 9:
            return 5, (-5 * DELTA)
        # if action == 10:
        #     return 6, (5 * DELTA)
        # if action == 11:
        #     return 6, (-5 * DELTA)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
