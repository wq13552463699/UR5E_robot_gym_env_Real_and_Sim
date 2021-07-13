#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 07:52:29 2021
@author: wq
"""

"""
This function need to be re-modified, because the application is tough, names of varibles are hard to understand.
"""

# Set TCP:
# You need to enter the TCP(dummy you added) position wrt to the UR5_connection!!! You can check the UR5_connection in the UR5's structure tree
# in coppeliasim.
# You just need to re-write the following variable, e.g. tcp_distance = 0.03
##########Only####Place####You####Need####To####Modify#####################################################
tcp_distance = 0.03
###########################################################################################################


pos = tcp_distance
import math
import numpy


#UR5 parmas
d1 = 0.089159
a2 = -0.42500
a3 = -0.39225
d4 = 0.10915
d5 = 0.09465
d6 = 0.0823 - 0.0064 + pos
PI = math.pi
ZERO_THRESH = 0.00000001
class Kinematic():
    def __init__(self):
        global d1
        global a2
        global a3
        global d4
        global d5
        global d6
        global PI
        global ZERO_THRESH

    def Forward(self,q):
        T=[0]*16
        #00
        s1 = math.sin(q[0])
        c1 = math.cos(q[0])
        ##01
        q234 = q[1]
        s2 = math.sin(q[1])
        c2 = math.cos(q[1])
        ##02
        s3 = math.sin(q[2])
        c3 = math.cos(q[2])
        q234 += q[2]
        ##03
        q234 += q[3]
        ##04
        s5 = math.sin(q[4])
        c5 = math.cos(q[4])
        ##05
        s6 = math.sin(q[5])
        c6 = math.cos(q[5])
        s234 = math.sin(q234)
        c234 = math.cos(q234)
        #4*4 roate arrary
        #00 parameter
        T[0] = ((c1 * c234 - s1 * s234) * s5) / 2.0 - c5 * s1 + ((c1 * c234 + s1 * s234) * s5) / 2.0
        # 01 parameter
        T[1] = (c6 * (s1 * s5 + ((c1 * c234 - s1 * s234) * c5) / 2.0 + ((c1 * c234 + s1 * s234) * c5) / 2.0) -
              (s6 * ((s1 * c234 + c1 * s234) - (s1 * c234 - c1 * s234))) / 2.0)
        # 02 parameter
        T[2] = (-(c6 * ((s1 * c234 + c1 * s234) - (s1 * c234 - c1 * s234))) / 2.0 -
              s6 * (s1 * s5 + ((c1 * c234 - s1 * s234) * c5) / 2.0 + ((c1 * c234 + s1 * s234) * c5) / 2.0))
        # 03 parameter
        T[3] = ((d5 * (s1 * c234 - c1 * s234)) / 2.0 - (d5 * (s1 * c234 + c1 * s234)) / 2.0 -
              d4 * s1 + (d6 * (c1 * c234 - s1 * s234) * s5) / 2.0 + (d6 * (c1 * c234 + s1 * s234) * s5) / 2.0 -
              a2 * c1 * c2 - d6 * c5 * s1 - a3 * c1 * c2 * c3 + a3 * c1 * s2 * s3)
        # 04 parameter
        T[4] = c1 * c5 + ((s1 * c234 + c1 * s234) * s5) / 2.0 + ((s1 * c234 - c1 * s234) * s5) / 2.0
        # 05 parameter
        T[5] = (c6 * (((s1 * c234 + c1 * s234) * c5) / 2.0 - c1 * s5 + ((s1 * c234 - c1 * s234) * c5) / 2.0) +
              s6 * ((c1 * c234 - s1 * s234) / 2.0 - (c1 * c234 + s1 * s234) / 2.0))
        # 06 parameter
        T[6] = (c6 * ((c1 * c234 - s1 * s234) / 2.0 - (c1 * c234 + s1 * s234) / 2.0) -
              s6 * (((s1 * c234 + c1 * s234) * c5) / 2.0 - c1 * s5 + ((s1 * c234 - c1 * s234) * c5) / 2.0))
        # 07 parameter
        T[7] = ((d5 * (c1 * c234 - s1 * s234)) / 2.0 - (d5 * (c1 * c234 + s1 * s234)) / 2.0 + d4 * c1 +
              (d6 * (s1 * c234 + c1 * s234) * s5) / 2.0 + (d6 * (s1 * c234 - c1 * s234) * s5) / 2.0 + d6 * c1 * c5 -
              a2 * c2 * s1 - a3 * c2 * c3 * s1 + a3 * s1 * s2 * s3)
        # 08 parameter
        T[8] = ((c234 * c5 - s234 * s5) / 2.0 - (c234 * c5 + s234 * s5) / 2.0)
        # 09 parameter
        T[9] = ((s234 * c6 - c234 * s6) / 2.0 - (s234 * c6 + c234 * s6) / 2.0 - s234 * c5 * c6)
        # 10 parameter
        T[10] = (s234 * c5 * s6 - (c234 * c6 + s234 * s6) / 2.0 - (c234 * c6 - s234 * s6) / 2.0)
        # 11 parameter
        T[11] = (d1 + (d6 * (c234 * c5 - s234 * s5)) / 2.0 + a3 * (s2 * c3 + c2 * s3) + a2 * s2 -
              (d6 * (c234 * c5 + s234 * s5)) / 2.0 - d5 * c234)
        # 12 parameter
        T[12] = 0.0
        # 13 parameter
        T[13] = 0.0
        # 14 parameter
        T[14] = 0.0
        # 15 parameter
        T[15] = 1.0
        return T



    def SIGN(self,x):
        return (x>0)-(x<0)
    def Forward_all(self,q):
        #result
        result=[]
        T1=[]
        T2=[]
        T3=[]
        T4=[]
        T5=[]
        T6=[]
        #q01
        s1 = math.sin(q[0])
        c1 = math.cos(q[0])
        #q02
        q23 = q[1]
        q234 = q[1]
        s2 = math.sin(q[1])
        c2 = math.cos(q[1])
        #q03
        s3 = math.sin(q[2])
        c3 = math.cos(q[2])
        q23 += q[2]
        q234 += q[2]
        #q04
        q234 += q[3]
        #q05
        s5 = math.sin(q[4])
        c5 = math.cos(q[4])
        #q06
        s6 = math.sin(q[5])
        c6 = math.cos(q[5])
        s23 = math.sin(q23)
        c23 = math.cos(q23)
        s234 = math.sin(q234)
        c234 = math.cos(q234)
        #01T
        #00-15 param
        T1[0] = c1
        T1[1] = 0
        T1[2] = s1
        T1[3] = 0
        T1[4] = s1
        T1[5] = 0
        T1[6] = -c1
        T1[7] = 0
        T1[8] = 0
        T1[9] = 1
        T1[10] = 0
        T1[11] =d1
        T1[12] = 0
        T1[13] = 0
        T1[14] = 0
        T1[15] = 1
        result.append(("T1",T1))
        #01T*12T=02T
        #00-15 parameters
        T2[0] = c1 * c2
        T2[1] = -c1 * s2
        T2[2] = s1
        T2[3] =a2 * c1 * c2
        T2[4] = c2 * s1
        T2[5] = -s1 * s2
        T2[6] = -c1
        T2[7] =a2 * c2 * s1
        T2[8] = s2
        T2[9] = c2
        T2[10] = 0
        T2[11] = d1 + a2 * s2
        T2[12] = 0
        T2[13] = 0
        T2[14] = 0
        T2[15] = 1
        result.append(("T2",T2))
        #01T*12T*23T=03T
        T3[0] = c23 * c1
        T3[1] = -s23 * c1
        T3[2] = s1
        T3[3] =c1 * (a3 * c23 + a2 * c2)
        T3[4] = c23 * s1
        T3[5] = -s23 * s1
        T3[6] = -c1
        T3[7] =s1 * (a3 * c23 + a2 * c2)
        T3[8] = s23
        T3[9] = c23
        T3[10] = 0
        T3[11] = d1 + a3 * s23 + a2 * s2
        T3[12] = 0
        T3[13] = 0
        T3[14] = 0
        T3[15] =1
        result.append(("T3",T3))
        #01T*12T*23T*34T=04T
        T4[0] = c234 * c1
        T4[1] = s1
        T4[2] = s234 * c1
        T4[3] =c1 * (a3 * c23 + a2 * c2) + d4 * s1
        T4[4] = c234 * s1
        T4[5] = -c1
        T4[6] = s234 * s1
        T4[7] =s1 * (a3 * c23 + a2 * c2) - d4 * c1
        T4[8] = s234
        T4[9] = 0
        T4[10] = -c234
        T4[11] =d1 + a3 * s23 + a2 * s2
        T4[12] =0
        T4[13] = 0
        T4[14] = 0
        T4[15] = 1
        result.append(("T4",T4))
        #01T*12T*23T*34T*45T=05T
        T5[0] = s1 * s5 + c234 * c1 * c5
        T5[1] = -s234 * c1
        T5[2] = c5 * s1 - c234 * c1 * s5
        T5[3] =c1 * (a3 * c23 + a2 * c2) + d4 * s1 + d5 * s234 * c1
        T5[4] = c234 * c5 * s1 - c1 * s5
        T5[5] = -s234 * s1
        T5[6] = - c1 * c5 - c234 * s1 * s5
        T5[7] =s1 * (a3 * c23 + a2 * c2) - d4 * c1 + d5 * s234 * s1
        T5[8] =  s234 * c5
        T5[9] = c234
        T5[10] = -s234 * s5
        T5[11] = d1 + a3 * s23 + a2 * s2 - d5 * c234
        T5[12] = 0
        T5[13] = 0
        T5[14] = 0
        T5[15] =1
        result.append(("T5",T5))
        #01T*12T*23T*34T*45T*56T=06T
        T6[0] =   c6 * (s1 * s5 + c234 * c1 * c5) - s234 * c1 * s6
        T6[1] = - s6 * (s1 * s5 + c234 * c1 * c5) - s234 * c1 * c6
        T6[2] = c5 * s1 - c234 * c1 * s5
        T6[3] =d6 * (c5 * s1 - c234 * c1 * s5) + c1 * (a3 * c23 + a2 * c2) + d4 * s1 + d5 * s234 * c1
        T6[4] = - c6 * (c1 * s5 - c234 * c5 * s1) - s234 * s1 * s6
        T6[5] = s6 * (c1 * s5 - c234 * c5 * s1) - s234 * c6 * s1
        T6[6] = - c1 * c5 - c234 * s1 * s5
        T6[7] =s1 * (a3 * c23 + a2 * c2) - d4 * c1 - d6 * (c1 * c5 + c234 * s1 * s5) + d5 * s234 * s1
        T6[8] =c234 * s6 + s234 * c5 * c6
        T6[9] = c234 * c6 - s234 * c5 * s6
        T6[10] = -s234 * s5
        T6[11] = d1 + a3 * s23 + a2 * s2 - d5 * c234 - d6 * s234 * s5
        T6[12] = 0
        T6[13] = 0
        T6[14] = 0
        T6[15] = 1
        result.append(("T6", T6))
        return result
    def Iverse(self,T,q6_des):
        q_sols=[0]*48
        num_sols = 0
        T02 = - T[0]
        T00 = T[1]
        T01 = T[2]
        T03 = -T[3]
        T12 = - T[4]
        T10 = T[5]
        T11 = T[6]
        T13 = - T[7]
        T22 = T[8]
        T20 = - T[9]
        T21 = - T[10]
        T23 = T[11]

        #####shoulder rotate  joint(q1) ###########################
        q1=[0,0]
        A = d6 * T12 - T13
        B = d6 * T02 - T03
        R = A * A + B * B
        if math.fabs(A) < ZERO_THRESH:
            if math.fabs(math.fabs(d4) - math.fabs(B)) < ZERO_THRESH:
                div = -self.SIGN(d4) * self.SIGN(B)
            else:
                div = -d4 / B
            arcsin = math.asin(div)
            if math.fabs(arcsin) < ZERO_THRESH:
                arcsin = 0.0
            if arcsin < 0.0:
                q1[0] = arcsin + 2.0 * PI
            else:
                q1[0] = arcsin
                q1[1] = PI - arcsin
        elif math.fabs(B) < ZERO_THRESH:
            if math.fabs(math.fabs(d4) - math.fabs(A)) < ZERO_THRESH:
                div = self.SIGN(d4) * self.SIGN(A)
            else:
                div = d4 / A
            arccos = math.acos(div)
            q1[0] = arccos
            q1[1] = 2.0 * PI - arccos
        elif d4 * d4 > R:
            return num_sols
        else:
            arccos = math.acos(d4 / math.sqrt(R))
            arctan = math.atan2(-B, A)
            pos = arccos + arctan
            neg = -arccos + arctan
            if math.fabs(pos) < ZERO_THRESH:
                pos = 0.0
            if math.fabs(neg) < ZERO_THRESH:
                neg = 0.0
            if pos >= 0.0:
                q1[0] = pos
            else:
                q1[0] = 2.0 * PI + pos
            if neg >= 0.0:
                q1[1] = neg
            else:
                q1[1] = 2.0 * PI + neg

        ###### wrist2  joint(q5) ##########################
        q5=numpy.zeros((2,2))#define 2*2 q5 array

        for i in range(2):
            numer = (T03 * math.sin(q1[i]) - T13 * math.cos(q1[i]) - d4)
            if math.fabs(math.fabs(numer) - math.fabs(d6)) < ZERO_THRESH:
                div = self.SIGN(numer) * self.SIGN(d6)
            else:
                div = numer / d6
            arccos = math.acos(div)
            q5[i][0] = arccos
            q5[i][1] = 2.0 * PI - arccos
        #############################################################
        for i in range(2):
            for j in range(2):
                c1 = math.cos(q1[i])
                s1 = math.sin(q1[i])
                c5 = math.cos(q5[i][j])
                s5 = math.sin(q5[i][j])
                ######################## wrist 3 joint (q6) ################################
                if math.fabs(s5) < ZERO_THRESH:
                    q6 = q6_des
                else:
                    q6 = math.atan2(self.SIGN(s5) * -(T01 * s1 - T11 * c1),self.SIGN(s5) * (T00 * s1 - T10 * c1))
                    if math.fabs(q6) < ZERO_THRESH:
                        q6 = 0.0
                    if (q6 < 0.0):
                        q6 += 2.0 * PI
                q2=[0,0]
                q3=[0,0]
                q4=[0,0]
                #####################RRR joints (q2, q3, q4) ################################
                c6 = math.cos(q6)
                s6 = math.sin(q6)
                x04x = -s5 * (T02 * c1 + T12 * s1) - c5 * (s6 * (T01 * c1 + T11 * s1) - c6 * (T00 * c1 + T10 * s1))
                x04y = c5 * (T20 * c6 - T21 * s6) - T22 * s5
                p13x = d5 * (s6 * (T00 * c1 + T10 * s1) + c6 * (T01 * c1 + T11 * s1)) - d6 * (T02 * c1 + T12 * s1) +T03 * c1 + T13 * s1
                p13y = T23 - d1 - d6 * T22 + d5 * (T21 * c6 + T20 * s6)
                c3 = (p13x * p13x + p13y * p13y - a2 * a2 - a3 * a3) / (2.0 * a2 * a3)
                if math.fabs(math.fabs(c3) - 1.0) < ZERO_THRESH:
                    c3 = self.SIGN(c3)
                elif math.fabs(c3) > 1.0:
                    # TODO NO SOLUTION
                    continue
                arccos = math.acos(c3)
                q3[0] = arccos
                q3[1] = 2.0 * PI - arccos
                denom = a2 * a2 + a3 * a3 + 2 * a2 * a3 * c3
                s3 = math.sin(arccos)
                A = (a2 + a3 * c3)
                B = a3 * s3
                q2[0] = math.atan2((A * p13y - B * p13x) / denom, (A * p13x + B * p13y) / denom)
                q2[1] = math.atan2((A * p13y + B * p13x) / denom, (A * p13x - B * p13y) / denom)
                c23_0 = math.cos(q2[0] + q3[0])
                s23_0 = math.sin(q2[0] + q3[0])
                c23_1 = math.cos(q2[1] + q3[1])
                s23_1 = math.sin(q2[1] + q3[1])
                q4[0] = math.atan2(c23_0 * x04y - s23_0 * x04x, x04x * c23_0 + x04y * s23_0)
                q4[1] = math.atan2(c23_1 * x04y - s23_1 * x04x, x04x * c23_1 + x04y * s23_1)
                ###########################################
                for k in range(2):
                    if math.fabs(q2[k]) < ZERO_THRESH:
                        q2[k] = 0.0
                    elif q2[k] < 0.0:
                        q2[k] += 2.0 * PI
                    if math.fabs(q4[k]) < ZERO_THRESH:
                        q4[k] = 0.0
                    elif q4[k] < 0.0:
                        q4[k] += 2.0 * PI
                    q_sols[num_sols * 6 + 0] = q1[i]
                    q_sols[num_sols * 6 + 1] = q2[k]
                    q_sols[num_sols * 6 + 2] = q3[k]
                    q_sols[num_sols * 6 + 3] = q4[k]
                    q_sols[num_sols * 6 + 4] = q5[i][j]
                    q_sols[num_sols * 6 + 5] = q6
                    num_sols+=1
        return num_sols,q_sols
    """
    q_inverse:ur kinematics solution joint q
    q_last:last joint q
    """
    def solve_2pi_pro(self,q_inverse,q_last):
        if abs(q_inverse-PI*2.0-q_last)<abs(q_inverse-q_last):
            q_inverse-=PI*2.0
            return q_inverse
        else:
            return q_inverse
    #ik_fk,1:ik,0:fk
    def display(self,q,q6_des,ik_fk):
        T = self.Forward(q)
        result_list=[]
        #q_sols=[0]*48
        num_sols, q_sols = self.Iverse(T, q6_des)
        j=0
        if ik_fk==1:
            for i in range(num_sols):
                print(i+1,"th solution")
                print( q_sols[j],q_sols[j+1],q_sols[j+2],q_sols[j+3],q_sols[j+4],q_sols[j+5])
                print( "solve 2 pi pro")
                print( self.solve_2pi_pro(q_sols[j],q[0]),self.solve_2pi_pro(q_sols[j+1],q[1]),self.solve_2pi_pro(q_sols[j+2],q[2]),self.solve_2pi_pro(q_sols[j+3],q[3]),\
                    self.solve_2pi_pro(q_sols[j+4], q[4]),self.solve_2pi_pro(q_sols[j+5],q[5]))
                j+=6
            print( "you have ",num_sols,"solutions")
        elif ik_fk==0:
            for i in range(4):
                print (T[j],T[j+1],T[j+2],T[j+3])
                j+=4
        else:
            print( "######################forward solution######################")
            for i in range(4):
                print( T[j],T[j+1],T[j+2],T[j+3])
                j+=4
            print( "######################inverse solutions#####################")
            j=0
            for i in range(num_sols):
                print (i+1,"th solution")
                print (q_sols[j],q_sols[j+1],q_sols[j+2],q_sols[j+3],q_sols[j+4],q_sols[j+5])
                result_list.append([q_sols[j],q_sols[j+1],q_sols[j+2],q_sols[j+3],q_sols[j+4],q_sols[j+5]])
                j+=6
            print( result_list)
            return result_list
            print ("you have ",num_sols,"solutions")
    """
    q:last time joint q
    """
    def get_ik_data(self,T,q):
        #T = self.Forward()
        result_list=[]
        #q_sols=[0]*48
        # print "T",T
        num_sols, q_sols = self.Iverse(T, 0)
        # print "num_sols, q_sols",num_sols, q_sols
        j = 0
        for i in range(num_sols):
            result_list.append([self.solve_2pi_pro(q_sols[j],q[0]),self.solve_2pi_pro(q_sols[j+1],q[1]),self.solve_2pi_pro(q_sols[j+2],q[2]),self.solve_2pi_pro(q_sols[j+3],q[3]),\
                    self.solve_2pi_pro(q_sols[j+4], q[4]),self.solve_2pi_pro(q_sols[j+5],q[5])])
            j += 6
        # print result_list
        return result_list

    def best_sol(self,weights,q_guess,T):
        #q_guess=q
        sols=self.get_ik_data(T,q_guess)
        # print "sols",sols
        valid_sols = []
        for sol in sols:
            test_sol = numpy.ones(6) * 9999.
            for i in range(6):
                for add_ang in [-2. * numpy.pi, 0, 2. * numpy.pi]:
                    test_ang = sol[i] + add_ang
                    if (abs(test_ang) <= 2. * numpy.pi and
                            abs(test_ang - q_guess[i]) < abs(test_sol[i] - q_guess[i])):
                        test_sol[i] = test_ang
            if numpy.all(test_sol != 9999.):
                valid_sols.append(test_sol)
        if len(valid_sols) == 0:
            return None
        best_sol_ind = numpy.argmin(numpy.sum((weights * (valid_sols - numpy.array(q_guess))) ** 2, 1))
        # print( "#########################the best sol##################")
        # print( valid_sols[best_sol_ind])
        return valid_sols[best_sol_ind]

    def best_sol_for_other_py(self,weights,q_guess,T):
        sols=self.get_ik_data(T)
        valid_sols = []
        for sol in sols:
            test_sol = numpy.ones(6) * 9999.
            for i in range(6):
                for add_ang in [-2. * numpy.pi, 0, 2. * numpy.pi]:
                    test_ang = sol[i] + add_ang
                    if (abs(test_ang) <= 2. * numpy.pi and
                            abs(test_ang - q_guess[i]) < abs(test_sol[i] - q_guess[i])):
                        test_sol[i] = test_ang
            if numpy.all(test_sol != 9999.):
                valid_sols.append(test_sol)
        if len(valid_sols) == 0:
            return None
        best_sol_ind = numpy.argmin(numpy.sum((weights * (valid_sols - numpy.array(q_guess))) ** 2, 1))
        print ("#########################the best sol##################")
        print (valid_sols[best_sol_ind].tolist())
        #print best_sol_ind
        print (sols[best_sol_ind])
        return valid_sols[best_sol_ind].tolist()