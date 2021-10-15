#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 16:39:27 2021

@author: qiang
"""

import sim
import math
import time

# TODO: These API function still need to be improved and modified. Some variables' defination are not uniformed. For example: position & pos
# TODO: Set raising error system to raise error for some condition. For example when the simulation is not start but users want to movej

joint_angle = [0, 0, 0, 0, 0, 0]  # each angle of joint
RAD2DEG = 180 / math.pi  # transform radian to degrees

class CoppeliaSim_env():
    def __init__(self,
                 scene_path = None,
                 scene_side = None,
                 commThreadCycleInMs = 5,
                 timeOutInMs = 5000):

        self.timeOutInMs = timeOutInMs
        self.commThreadCycleInMs = commThreadCycleInMs
        self.scene_path = scene_path
        self.scene_side = scene_side
        self.sim_activate()
        
        if self.scene_path:
            assert self.scene_path and self.scene_side, "Miss the Argument 'scene_side', you must specify which side the file is located"
            sim.simxLoadScene(self.clientID,self.scene_path,self.scene_side,sim.simx_opmode_blocking)
            
        self.sim_is_running = 0
        
    
    def sim_activate(self):
        sim.simxFinish(-1)
        # Close the potential connection
        while True:
            # simxStart的参数分别为：服务端IP地址(连接本机用127.0.0.1);端口号;是否等待服务端开启;连接丢失时是否尝试再次连接;超时时间(ms);数据传输间隔(越小越快)
            self.clientID = sim.simxStart('127.0.0.1', 19998, True, True, self.timeOutInMs, self.commThreadCycleInMs)
            if self.clientID > -1:
                print("Connection success!")
                break
            else:
                time.sleep(0.2)
                print("Failed connecting to remote API server!")
                print("Maybe you forget to open CoppliaSim, please check")
        self.sim_is_activate = 1
        
    def sim_finish(self):
        if self.sim_is_running:
            self.sim_stop()
            time.sleep(0.5)
        sim.simxFinish(self.clientID)
        self.sim_is_activate = 0
    
    def sim_start(self):
        if self.sim_is_activate:
            sim.simxStartSimulation(self.clientID, sim.simx_opmode_oneshot)
            self.sim_is_running = 1
        else:
            print('You need to activate the simulation before start it')
            
    def sim_pause(self):
        if self.sim_is_activate:
            if self.sim_is_running:
                sim.simxPauseSimulation(self.clientID, sim.simx_opmode_oneshot)
                self.sim_is_running = 0
            else:
                print('Simulation was not started')
        else:
            print('You need to activate the simulation before pause it')
            
    def sim_stop(self):
        if self.sim_is_activate:
            if self.sim_is_running:
                sim.simxStopSimulation(self.clientID, sim.simx_opmode_oneshot)
                self.sim_is_running = 0
            else:
                print('Simulation was not started')
        else:
            print('You need to activate the simulation before stop it')
            
    def sim_recover(self):
        if self.sim_is_activate:
            if self.sim_is_running:
                sim.simxStopSimulation(self.clientID, sim.simx_opmode_oneshot)
                time.sleep(0.5)
                sim.simxStartSimulation(self.clientID, sim.simx_opmode_oneshot)
            else:
                print('Simulation was not started')
        else:
            print('You need to activate the simulation before recover it')
    

    
