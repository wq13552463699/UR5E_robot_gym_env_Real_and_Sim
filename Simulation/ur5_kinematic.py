'''
This class include all kinematic function
You need to set the position of the robot, you can check it in coppeliasim. The default position is [0.0,0.0,0.0]
Users must aware: The orientation of the robot must be: Alpha:0 Beta:0 Gamma:-90. For current API, we don't have the
orientation transformation, so you need to put your robot like above orientation by yourself in advance and cannot change
it.
'''
from math import pi
from math_cal import Pose_2_UR
from rl_utils import list2mat
from kinematic import Kinematic
# init_pos = [0.0,0.0,0.0145]

init_pos = [0.0,0.0,0.0]
class UR5_kinematic(Kinematic):
    def __init__(self, UR5_global_position=[0.0,0.0,0.0]):
        self.UR5_global_position = [UR5_global_position[0] - init_pos[0],UR5_global_position[1] - init_pos[1],
                                    UR5_global_position[2] - init_pos[2]]

    def Forward_ur5(self,joint_angles):
        ref = [0,-pi/2,0,-pi/2,0,0]
        actual_joint_angles = [joint_angles[0]+ref[0],joint_angles[1]+ref[1],joint_angles[2]+ref[2],joint_angles[3]+
                               ref[3],joint_angles[4]+ref[4],joint_angles[5]+ref[5]]
        pos_list = self.Forward(actual_joint_angles)
        pos = list2mat(pos_list)
        UR_local = Pose_2_UR(pos)
        UR_global = [UR_local[0]+self.UR5_global_position[0],UR_local[1]+self.UR5_global_position[1],UR_local[2]
                     +self.UR5_global_position[2],UR_local[3],UR_local[4],UR_local[5]]
        return UR_global