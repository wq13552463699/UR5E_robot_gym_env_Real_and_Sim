# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 16:12:19 2021

@author: 14488
"""
import numpy as np
from math import sin,cos,pi

# + 0.0787866
############################# Constant ###################################
# p0_1 = 0.089159 
# p1_2 = 0.13585
# p2_3 = 0.425
# p3_5 = 0.1197
# p5_7 = 0.39225
# p7_9 = 0.093
# p9_11 = 0.09465
# p11_13 = 0.0823

p0_1 = 0.1625
p2_3 = 0.425
p5_7 = 0.3922

p1_2 = 0.13585

p7_9 = 0.11715

p9_11 = 0.0997
p11_13 = 0.0996

p3_5 = 0.1197
p3_4 = 0.07115
p5_6 = 0.04215
p7_8 = 0.05915
p9_10 = 0.05741
p11_12 = 0.05715
p13_14 = 0.01055


r_s1 = 0.120 / 2
r_s2 = 0.120 / 2
r_s3 = 0.0776 / 2
r_s4 = 0.08 / 2
r_s5 = 0.078 / 2
r_s6 = 0.075 / 2

rela_tip_pos = 0#0.0295 #+ 0.012601861
tip_pos = rela_tip_pos + p11_13
##########################################################################

############################# Functional part ############################
class UR5e_kin():
    def __init__(self,
                 mode=24,
                 global_init = [-pi/2,-pi/2,0,-pi/2,0,0]):
        if mode == 24:
            self.d1,self.d2,self.d3,self.d4,self.d5,self.d6,self.d7,self.d8,self.d9,self.d10,self.d11,self.d12,self.d13,self.d14,self.d15,self.d16,self.d17,self.d18,self.d19,self.d20,self.d21,self.d22,\
            self.d23,self.d24,self.tip = self.Get24_local_coor()
        if mode == 48:
            self.d1,self.d2,self.d3,self.d4,self.d5,self.d6,self.d7,self.d8,self.d9,self.d10,self.d11,self.d12,self.d13,self.d14,self.d15,self.d16,self.d17,self.d18,self.d19,self.d20,self.d21,self.d22,\
                self.d23,self.d24,self.d25,self.d26,self.d27,self.d28,self.d29,self.d30,self.d31,self.d32,self.d33,self.d34,self.d35,self.d36,self.d37,self.d38,self.d39,self.d40,\
                    self.d41,self.d42,self.d43,self.d44,self.d45,self.d46,self.d47,self.d48,self.tip = self.Get48_local_coor()
    
        self.global_init = global_init
    
    def Trans(self,axis,dis):
        if axis == 'x':
            return np.mat([[1,0,0,dis],
                           [0,1,0,0],
                           [0,0,1,0],
                           [0,0,0,1]])
        if axis == 'y':
            return np.mat([[1,0,0,0],
                           [0,1,0,dis],
                           [0,0,1,0],
                           [0,0,0,1]])
        if axis == 'z':
            return np.mat([[1,0,0,0],
                           [0,1,0,0],
                           [0,0,1,dis],
                           [0,0,0,1]])
    def Rot(self,axis,angle):
        if axis == 'x':
            return np.mat([[1,0,0,0],
                           [0,cos(angle),-sin(angle),0],
                           [0,sin(angle),cos(angle),0],
                           [0,0,0,1]])
        if axis == 'y':
            return np.mat([[cos(angle),0,sin(angle),0],
                           [0,1,0,0],
                           [-sin(angle),0,cos(angle),0],
                           [0,0,0,1]])
        if axis == 'z':
            return np.mat([[cos(angle),-sin(angle),0,0],
                           [sin(angle),cos(angle),0,0],
                           [0,0,1,0],
                           [0,0,0,1]])
        
    def coor(self,coord):
        return np.mat([coord[0],coord[1],coord[2],1]).T
    
    def D4_generator(self,S):
        if S == 's1':
            d1 = self.coor([-p3_4,0,r_s1])
            d2 = self.coor([-p3_4,-r_s1,0])
            d3 = self.coor([-p3_4,0,-r_s1])
            d4 = self.coor([-p3_4,r_s1,0])
        elif S == 's2':
            d1 = self.coor([p5_6,0,r_s2])
            d2 = self.coor([p5_6,r_s2,0])
            d3 = self.coor([p5_6,0,-r_s2])
            d4 = self.coor([p5_6,-r_s2,0])
        elif S == 's3':
            d1 = self.coor([p7_8,0,r_s3])
            d2 = self.coor([p7_8,r_s3,0])
            d3 = self.coor([p7_8,0,-r_s3])
            d4 = self.coor([p7_8,-r_s3,0])
        elif S == 's4':
            d1 = self.coor([-r_s4,0,-p9_10])
            d2 = self.coor([0,-r_s4,-p9_10])
            d3 = self.coor([r_s4,0,-p9_10])
            d4 = self.coor([0,r_s4,-p9_10])
        elif S == 's5':
            d1 = self.coor([p11_12,0,r_s5])
            d2 = self.coor([p11_12,r_s5,0])
            d3 = self.coor([p11_12,0,-r_s5])
            d4 = self.coor([p11_12,-r_s5,0])
        if S == 's6':
            d1 = self.coor([-p11_13-p13_14,0,r_s6])
            d2 = self.coor([-p11_13-p13_14,-r_s6,0])
            d3 = self.coor([-p11_13-p13_14,0,-r_s6])
            d4 = self.coor([-p11_13-p13_14,r_s6,0])
        return d1,d2,d3,d4
            
    def D8_generator(self,S):
        d1,d3,d5,d7 = self.D4_generator(S)
        if S == 's1':
            d2 = self.coor([-p3_4,-r_s1*sin(pi/4),r_s1*sin(pi/4)])
            d4 = self.coor([-p3_4,-r_s1*sin(pi/4),-r_s1*sin(pi/4)])
            d6 = self.coor([-p3_4,r_s1*sin(pi/4),-r_s1*sin(pi/4)])
            d8 = self.coor([-p3_4,r_s1*sin(pi/4),r_s1*sin(pi/4)])
        elif S == 's2':
            d2 = self.coor([p5_6,r_s2*sin(pi/4),r_s2*sin(pi/4)])
            d4 = self.coor([p5_6,r_s2*sin(pi/4),-r_s2*sin(pi/4)])
            d6 = self.coor([p5_6,-r_s2*sin(pi/4),-r_s2*sin(pi/4)])
            d8 = self.coor([p5_6,-r_s2*sin(pi/4),r_s2*sin(pi/4)])
        elif S == 's3':
            d2 = self.coor([p7_8,r_s3*sin(pi/4),r_s3*sin(pi/4)])
            d4 = self.coor([p7_8,r_s3*sin(pi/4),-r_s3*sin(pi/4)])
            d6 = self.coor([p7_8,-r_s3*sin(pi/4),-r_s3*sin(pi/4)])
            d8 = self.coor([p7_8,-r_s3*sin(pi/4),r_s3*sin(pi/4)])
        elif S == 's4':
            d2 = self.coor([-r_s4*sin(pi/4),-r_s4*sin(pi/4),-p9_10])
            d4 = self.coor([r_s4*sin(pi/4),-r_s4*sin(pi/4),-p9_10])
            d6 = self.coor([r_s4*sin(pi/4),r_s4*sin(pi/4),-p9_10])
            d8 = self.coor([-r_s4*sin(pi/4),r_s4*sin(pi/4),-p9_10])
        elif S == 's5':
            d2 = self.coor([p11_12,r_s5*sin(pi/4),r_s5*sin(pi/4)])
            d4 = self.coor([p11_12,r_s5*sin(pi/4),-r_s5*sin(pi/4)])
            d6 = self.coor([p11_12,-r_s5*sin(pi/4),-r_s5*sin(pi/4)])
            d8 = self.coor([p11_12,-r_s5*sin(pi/4),r_s5*sin(pi/4)])
        elif S == 's6':
            d2 = self.coor([-p11_13-p13_14,-r_s6*sin(pi/4),r_s6*sin(pi/4)])
            d4 = self.coor([-p11_13-p13_14,-r_s6*sin(pi/4),-r_s6*sin(pi/4)])
            d6 = self.coor([-p11_13-p13_14,r_s6*sin(pi/4),-r_s6*sin(pi/4)])
            d8 = self.coor([-p11_13-p13_14,r_s6*sin(pi/4),r_s6*sin(pi/4)])
        return d1,d2,d3,d4,d5,d6,d7,d8
    
    def Get24_local_coor(self):
        d1,d2,d3,d4 = self.D4_generator('s1')
        d5,d6,d7,d8 = self.D4_generator('s2')
        d9,d10,d11,d12 = self.D4_generator('s3')
        d13,d14,d15,d16 = self.D4_generator('s4')
        d17,d18,d19,d20 = self.D4_generator('s5')
        d21,d22,d23,d24 = self.D4_generator('s6')
        tip = self.coor([-tip_pos,0,0])
        return d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,\
            d21,d22,d23,d24,tip
        
    def Get48_local_coor(self):
        d1,d2,d3,d4,d5,d6,d7,d8 = self.D8_generator('s1')
        d9,d10,d11,d12,d13,d14,d15,d16 = self.D8_generator('s2')
        d17,d18,d19,d20,d21,d22,d23,d24 = self.D8_generator('s3')
        d25,d26,d27,d28,d29,d30,d31,d32 = self.D8_generator('s4')
        d33,d34,d35,d36,d37,d38,d39,d40 = self.D8_generator('s5')
        d41,d42,d43,d44,d45,d46,d47,d48 = self.D8_generator('s6')
        tip = self.coor([-tip_pos,0,0])
        return d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,\
            d21,d22,d23,d24,d25,d26,d27,d28,d29,d30,d31,d32,d33,d34,d35,d36,d37,d38,d39,d40,\
            d41,d42,d43,d44,d45,d46,d47,d48,tip
            
    ############################# Local coordinate ##############################
    
        
    
    ############################################################################
        
    def Get24_global_coor(self,joint_angles):
        s1_rot_mat = self.Rot('z',joint_angles[0]) * self.Trans('z', p0_1) * self.Trans('x', -p1_2) * self.Rot('x',-joint_angles[1]) * self.Trans('z', p2_3)
        gd1 = s1_rot_mat * self.d1
        gd2 = s1_rot_mat * self.d2
        gd3 = s1_rot_mat * self.d3
        gd4 = s1_rot_mat * self.d4
        
        s2_rot_mat = s1_rot_mat * self.Trans('x', p3_5)
        gd5 = s2_rot_mat * self.d5
        gd6 = s2_rot_mat * self.d6
        gd7 = s2_rot_mat * self.d7
        gd8 = s2_rot_mat * self.d8
        
        s3_rot_mat = s2_rot_mat * self.Rot('x',-joint_angles[2]) * self.Trans('z', p5_7)
        gd9 = s3_rot_mat * self.d9
        gd10 = s3_rot_mat * self.d10
        gd11 = s3_rot_mat * self.d11
        gd12 = s3_rot_mat * self.d12
        
        s4_rot_mat = s3_rot_mat * self.Trans('x', -p7_9) * self.Rot('x',-joint_angles[3])
        gd13 = s4_rot_mat * self.d13
        gd14 = s4_rot_mat * self.d14
        gd15 = s4_rot_mat * self.d15
        gd16 = s4_rot_mat * self.d16
        
        s5_rot_mat = s4_rot_mat * self.Trans('z', p9_11) * self.Rot('z',joint_angles[4])
        gd17 = s5_rot_mat * self.d17
        gd18 = s5_rot_mat * self.d18
        gd19 = s5_rot_mat * self.d19
        gd20 = s5_rot_mat * self.d20
        gd21 = s5_rot_mat * self.d21
        gd22 = s5_rot_mat * self.d22
        gd23 = s5_rot_mat * self.d23
        gd24 = s5_rot_mat * self.d24
        
        g_tip = s5_rot_mat * self.tip
        
        return [gd1,gd2,gd3,gd4,gd5,gd6,gd7,gd8,gd9,gd10,gd11,gd12,gd13,gd14,gd15,\
            gd16,gd17,gd18,gd19,gd20,gd21,gd22,gd23,gd24,g_tip]
            
    def Get48_global_coor(self,joint_angles):
        
        joint_angles[0] -= self.global_init[0]
        joint_angles[1] -= self.global_init[1]
        joint_angles[2] -= self.global_init[2]
        joint_angles[3] -= self.global_init[3]
        joint_angles[4] -= self.global_init[4]
        
        s1_rot_mat = self.Rot('z',joint_angles[0]) * self.Trans('z', p0_1) * self.Trans('x', -p1_2) * self.Rot('x',-joint_angles[1]) * self.Trans('z', p2_3)
        gd1 = s1_rot_mat * self.d1
        gd2 = s1_rot_mat * self.d2
        gd3 = s1_rot_mat * self.d3
        gd4 = s1_rot_mat * self.d4
        gd5 = s1_rot_mat * self.d5
        gd6 = s1_rot_mat * self.d6
        gd7 = s1_rot_mat * self.d7
        gd8 = s1_rot_mat * self.d8
        
        s2_rot_mat = s1_rot_mat * self.Trans('x', p3_5)
        gd9 = s2_rot_mat * self.d9
        gd10 = s2_rot_mat * self.d10
        gd11 = s2_rot_mat * self.d11
        gd12 = s2_rot_mat * self.d12
        gd13 = s2_rot_mat * self.d13
        gd14 = s2_rot_mat * self.d14
        gd15 = s2_rot_mat * self.d15
        gd16 = s2_rot_mat * self.d16
        
        s3_rot_mat = s2_rot_mat * self.Rot('x',-joint_angles[2]) * self.Trans('z', p5_7)
        gd17 = s3_rot_mat * self.d17
        gd18 = s3_rot_mat * self.d18
        gd19 = s3_rot_mat * self.d19
        gd20 = s3_rot_mat * self.d20
        gd21 = s3_rot_mat * self.d21
        gd22 = s3_rot_mat * self.d22
        gd23 = s3_rot_mat * self.d23
        gd24 = s3_rot_mat * self.d24
        
        s4_rot_mat = s3_rot_mat * self.Trans('x', -p7_9) * self.Rot('x',-joint_angles[3])
        gd25 = s4_rot_mat * self.d25
        gd26 = s4_rot_mat * self.d26
        gd27 = s4_rot_mat * self.d27
        gd28 = s4_rot_mat * self.d28
        gd29 = s4_rot_mat * self.d29
        gd30 = s4_rot_mat * self.d30
        gd31 = s4_rot_mat * self.d31
        gd32 = s4_rot_mat * self.d32
        
        s5_rot_mat = s4_rot_mat * self.Trans('z', p9_11) * self.Rot('z',joint_angles[4])
        gd33 = s5_rot_mat * self.d33
        gd34 = s5_rot_mat * self.d34
        gd35 = s5_rot_mat * self.d35
        gd36 = s5_rot_mat * self.d36
        gd37 = s5_rot_mat * self.d37
        gd38 = s5_rot_mat * self.d38
        gd39 = s5_rot_mat * self.d39
        gd40 = s5_rot_mat * self.d40
        gd41 = s5_rot_mat * self.d41
        gd42 = s5_rot_mat * self.d42
        gd43 = s5_rot_mat * self.d43
        gd44 = s5_rot_mat * self.d44
        gd45 = s5_rot_mat * self.d45
        gd46 = s5_rot_mat * self.d46
        gd47 = s5_rot_mat * self.d47
        gd48 = s5_rot_mat * self.d48
        
        g_tip = s5_rot_mat * self.tip
        
        mtx = [gd1,gd2,gd3,gd4,gd5,gd6,gd7,gd8,gd9,gd10,gd11,gd12,gd13,gd14,gd15,\
            gd16,gd17,gd18,gd19,gd20,gd21,gd22,gd23,gd24,gd25,gd26,gd27,gd28,gd29,\
            gd30,gd31,gd32,gd33,gd34,gd35,gd36,gd37,gd38,gd39,gd40,gd41,gd42,\
            gd43,gd44,gd45,gd46,gd47,gd48,g_tip]
        
        lst = []
        
        for i in mtx:
            lst.append(i.T.tolist())
        
        return lst
        