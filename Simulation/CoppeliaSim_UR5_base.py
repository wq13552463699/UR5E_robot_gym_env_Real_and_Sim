#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 17:19:12 2021

@author: qiang
"""

from CoppeliaSim_env import CoppeliaSim_env
from ur5_kinematic import UR5_kinematic
import sim
import numpy as np
from math import pi
from math_cal import UR_2_Pose
from rl_utils import *

# UR5_global_position = [0,0,0]
# ur5_safebox_high = [0.23,0.45,0.83]
# ur5_safebox_low = [-0.87,-0.4,0]
# target_range_low = [-0.72,-0.3,0.035]
# target_range_high = [-0.42,0.3,0.035]

UR5_global_position = [0.4,-0.425,0.65]
ur5_safebox_high = [0.72,0.275,1.65]
ur5_safebox_low = [-0.72,-1.025,0.65]
target_range_low = [-0.343,-0.737,0.685]
target_range_high = [0.041,-0.003,0.685]

class CoppeliaSim_UR5_base(CoppeliaSim_env):
    
    def __init__(self,
                 scene_path = None,
                 scene_side = None
                 ):
        
        super().__init__(scene_path=scene_path,scene_side=scene_side)
        self.baseName = 'UR5'
        self.joint1Name = 'UR5_joint1'
        self.joint2Name = 'UR5_joint2'
        self.joint3Name = 'UR5_joint3'
        self.joint4Name = 'UR5_joint4'
        self.joint5Name = 'UR5_joint5'
        self.joint6Name = 'UR5_joint6'
        self.origin = 'origin'
        self.tip = 'ur_tip'
        self.target = 'ur_target'
        self.kin = UR5_kinematic(UR5_global_position)
        
        _, self.joint1Handle = sim.simxGetObjectHandle(self.clientID, self.joint1Name, sim.simx_opmode_blocking)
        _, self.joint2Handle = sim.simxGetObjectHandle(self.clientID, self.joint2Name, sim.simx_opmode_blocking)
        _, self.joint3Handle = sim.simxGetObjectHandle(self.clientID, self.joint3Name, sim.simx_opmode_blocking)
        _, self.joint4Handle = sim.simxGetObjectHandle(self.clientID, self.joint4Name, sim.simx_opmode_blocking)
        _, self.joint5Handle = sim.simxGetObjectHandle(self.clientID, self.joint5Name, sim.simx_opmode_blocking)
        _, self.joint6Handle = sim.simxGetObjectHandle(self.clientID, self.joint6Name, sim.simx_opmode_blocking)
        _, self.baseHandle = sim.simxGetObjectHandle(self.clientID, self.baseName, sim.simx_opmode_blocking)
        _, self.originHandle = sim.simxGetObjectHandle(self.clientID, self.origin, sim.simx_opmode_blocking)
        _, self.above_cameraHandle = sim.simxGetObjectHandle(self.clientID, 'Above_camera', sim.simx_opmode_blocking)
        # print("initsssssssssssssssssss")
        _, self.side_camera2Handle = sim.simxGetObjectHandle(self.clientID, 'Side_camera2', sim.simx_opmode_blocking)
        _, self.side_camera3Handle = sim.simxGetObjectHandle(self.clientID, 'Side_camera3', sim.simx_opmode_blocking)
        _, self.side_camera1Handle = sim.simxGetObjectHandle(self.clientID, 'Side_camera1', sim.simx_opmode_blocking)
        _, self.tipHandle = sim.simxGetObjectHandle(self.clientID, self.tip, sim.simx_opmode_blocking)
        _, self.targetHandle = sim.simxGetObjectHandle(self.clientID, self.target, sim.simx_opmode_blocking)
        
        
    def get_joint_handle(self, jointNum):
        if jointNum == 1:
            return self.joint1Handle
        if jointNum == 2:
            return self.joint2Handle
        if jointNum == 3:
            return self.joint3Handle
        if jointNum == 4:
            return self.joint4Handle
        if jointNum == 5:
            return self.joint5Handle
        if jointNum == 6:
            return self.joint6Handle
        if jointNum == 7:
            return self.joint7Handle

    def get_joint_position(self, jointNum):
        # This function can be appiled to get a joint's position
        jointHandle = self.get_joint_handle(jointNum)
        clientID = self.clientID
        _, jpos = sim.simxGetJointPosition(clientID, jointHandle, sim.simx_opmode_blocking)
        return jpos

    def get_all_joints_position(self):
        # This function will get all joints posotions and return a list.
        p1 = self.get_joint_position(1)
        p2 = self.get_joint_position(2)
        p3 = self.get_joint_position(3)
        p4 = self.get_joint_position(4)
        p5 = self.get_joint_position(5)
        p6 = self.get_joint_position(6)
        return np.array([p1,p2,p3,p4,p5,p6])

    def get_tip_position(self):
        tipHandle = self.tipHandle
        clientID = self.clientID
        _, tip_xyz_pos = sim.simxGetObjectPosition(clientID,tipHandle,self.originHandle,sim.simx_opmode_blocking)
        return np.array(tip_xyz_pos)

    def get_target_position(self):
        targetHandle = self.targetHandle
        clientID = self.clientID
        _, target_xyz_pos = sim.simxGetObjectPosition(clientID,targetHandle,self.originHandle,sim.simx_opmode_blocking)
        return np.array(target_xyz_pos)

    def random_gen_one_target_pos(self):
        pos = np.random.uniform(target_range_low,target_range_high)
        return pos

    def random_gen_n_target_pos(self,size):
        pos = np.random.uniform(target_range_low,target_range_high,(size,3))
        return pos

    def set_target_to_random_pos(self):
        pos = self.random_gen_one_target_pos()
        self.set_target_pos(pos)
        return pos

    def set_target_pos(self,position):
        targetHandle = self.targetHandle
        clientID = self.clientID
        sim.simxSetObjectPosition(clientID, targetHandle, self.originHandle,position,sim.simx_opmode_blocking)

    def move_joint_angle(self, jointNum, angle):
        # Move a joint to certain angle wrt current joint's position
        cur_pos = self.get_joint_position(jointNum)
        clientID = self.clientID
        jointHandle = self.get_joint_handle(jointNum)
        sim.simxSetJointTargetPosition(clientID, jointHandle, (cur_pos + angle), sim.simx_opmode_oneshot)

    def movej(self,angles):
        # Movej function
        self.move_joint_to(1, angles[0])
        self.move_joint_to(2, angles[1])
        self.move_joint_to(3, angles[2])
        self.move_joint_to(4, angles[3])
        self.move_joint_to(5, angles[4])
        self.move_joint_to(6, angles[5])

    def move_joint_to(self, jointNum, pos):
        # Move a joint to a certain angle
        clientID = self.clientID
        jointHandle = self.get_joint_handle(jointNum)
        sim.simxSetJointTargetPosition(clientID, jointHandle, pos, sim.simx_opmode_oneshot)

    def movearm(self, pos_list):
        from math import pi
        init_pos = [0, pi / 2, 0, pi / 2, 0, 0]
        i = 1
        for pos in pos_list:
            self.move_joint_to(i, pos + init_pos[i - 1])
            i += 1

    def arm_move_to(self, ur_pos, ref_point=[0, -pi / 2, 0, -pi / 2, 0, 0]):
        '''
        :param ur_pos: UR5's style 6 values pose list
        :param ref_point: Because to move arm to a position, this function need to use the Inverse kinematic, so you need
                        to set a reference, the iverse kinematic function will find the best trajectory wrt the reference
                        point.
        '''
        pose = UR_2_Pose(ur_pos)
        pose_list = mat2list(pose)
        weights = [1.] * 6
        best_angles = self.kin.best_sol(weights, ref_point, pose_list)
        self.movearm(best_angles)

    def pos_after_move(self, angles):
        '''
        :param angles: A list include all 6 joints' angles needed to be moved.
        :return: The position of TCP after movement, which is in UR's 6 values style.
        '''
        ur_pos = self.kin.Forward_ur5(angles)
        # print("The position after movement is:")
        # print(ur_pos)
        return ur_pos

        # Control RG2 gripper
        # openRG2 and closeRG2 are 2 functions that can be applied to control the RG2 gripper.
        # To make these 2 fucntion valid, you need to add the following script into RG2's child scipt in coppleliasim
        '''
         rg2Close = function(inInts,inFloats,inStrings,inBuffer)
            local v = -motorVelocity
            sim.setJointForce(motorHandle,motorForce)
            sim.setJointTargetVelocity(motorHandle,v)
            return {},{v},{},''
        end

        rg2Open = function(inInts,inFloats,inStrings,inBuffer)
            local v = motorVelocity
            sim.setJointForce(motorHandle,motorForce)
            sim.setJointTargetVelocity(motorHandle,v)
            return {},{v},{},''
        end

        function sysCall_init( )
            motorHandle=sim.getObjectHandle('RG2_openCloseJoint')
            motorVelocity=0.05 -- m/s
            motorForce=20 -- N
            rg2Open({}, {}, {}, '')
        end
        '''
    def openRG2(self):
        rgName = self.rgName
        clientID = self.clientID
        res, retInts, retFloats, retStrings, retBuffer = sim.simxCallScriptFunction(clientID, rgName, \
                                                                                    sim.sim_scripttype_childscript,
                                                                                    'rg2Open', [], [], [], b'',
                                                                                    sim.simx_opmode_blocking)

    def closeRG2(self):
        rgName = self.rgName
        clientID = self.clientID
        res, retInts, retFloats, retStrings, retBuffer = sim.simxCallScriptFunction(clientID, rgName, \
                                                                                    sim.sim_scripttype_childscript,
                                                                                    'rg2Close', [], [], [], b'',
                                                                                    sim.simx_opmode_blocking)



    def get_image(self, camera_name):
        if camera_name == "Above_camera":
            cameraHandle = self.above_cameraHandle
        if camera_name == 'Side_camera2':
            cameraHandle = self.side_camera2Handle
        if camera_name == 'Side_camera1':
            cameraHandle = self.side_camera1Handle
        if camera_name == 'Side_camera3':
            cameraHandle = self.side_camera3Handle
            
        if self.sim_is_running == 1:
            sim_ret, resolution, image_rgb = sim.simxGetVisionSensorImage(self.clientID, cameraHandle, 0,
                                                                      sim.simx_opmode_blocking)
            result_rgb = brg2rgb(image_rgb, resolution)
            return result_rgb
            # return np.array(image_rgb)
        else:
            print('You need to start the simulation before get image')
        
        
        
        
        
        
        
        
        
        
        
        