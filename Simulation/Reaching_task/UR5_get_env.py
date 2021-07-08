# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 18:13:27 2021

@author: 14488
"""

from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.ur5 import UR5
from pyrep.objects.shape import Shape
import numpy as np
from math import pi,sqrt
#import os

DELTA = pi/180.0

SCENE_FILE = join(dirname(abspath(__file__)),
                  'houzhuohong.ttt')
POS_MIN, POS_MAX = [-0.37, -0.855, 0.7], [0.37, -0.395, 0.7]
EPISODES = 5
EPISODE_LENGTH = 200

import tensorflow as tf 
import random

#创造双向队列
from collections import deque
from random import choice
# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q 
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch
REACH_THRESHOLD = 0.01
ACTION_RANGE = [1,2,3,4,5,6,7,8,9,10,11,12]
MOVE_X1 = -0.65
MOVE_X2 = 0.65
MOVE_Y1 = -0.72
MOVE_Y2 = 0.72
MOVE_Z1 = 0.64


class ReacherEnv(object):

    def __init__(self):
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=False)
        self.pr.start()
        self.agent = UR5()
#        self.agent.set_control_loop_enabled(False)
#        self.agent.set_motor_locked_at_zero_velocity(True)
        self.target = Shape('target')
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()
        self.last_joint_positions = self.agent.get_joint_positions()
        self.state_dim = self._get_state
        
    def _get_obs_dim(self):
        state_example = self._get_state()
        return state_example.shape[0]
    
    def _get_action_dim(self, robot):
        if robot == 'UR5':
            return 12 #6*2
    
    def _get_state(self):
        # Return state containing arm joint angles/velocities & target position
        return np.concatenate([self.agent.get_joint_positions(),
                               self.agent_ee_tip.get_position(),
                               self.target.get_position()])

    def reset(self):
        # Get a random position within a cuboid and set the target position
        pos = list(np.random.uniform(POS_MIN, POS_MAX))
        self.target.set_position(pos)
        self.agent.set_joint_positions(self.initial_joint_positions)
        self.last_joint_positions = self.initial_joint_positions
        self.pr.step()
        return self._get_state()

    def step(self, action):
        move_joint, angle = self.action_space(action)
        self.last_joint_positions[move_joint-1] += angle
        self.agent.set_joint_positions(self.last_joint_positions)  # Execute action on arm
        self.pr.step()  # Step the physics simulation
        ax, ay, az = self.agent_ee_tip.get_position()
        tx, ty, tz = self.target.get_position()
        # Reward is negative distance to target
        reward = -np.sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)
        next_state =  self._get_state()
        done = self.check_done()
        return next_state,reward,done
    
    def check_done(self):
        ax, ay, az = self.agent_ee_tip.get_position()
        tx, ty, tz = self.target.get_position()
        if sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2) <= REACH_THRESHOLD:
            return 1
        elif ax <= MOVE_X1 or ax >= MOVE_X2:
            return 1
        elif ay <= MOVE_Y1 or ay >= MOVE_Y2:
            return 1
        elif az <= MOVE_Z1:
            return 1
        else:
            return 0
        
    def action_space(self,action):
        if action == 1:
            return 1,5*DELTA  #joint degree
        if action == 2:
            return 1,-5*DELTA
        if action == 3:
            return 2,5*DELTA
        if action == 4:
            return 2,-5*DELTA
        if action == 5:
            return 3,5*DELTA
        if action == 6:
            return 3,-5*DELTA
        if action == 7:
            return 4,5*DELTA
        if action == 8:
            return 4,-5*DELTA
        if action == 9:
            return 5,5*DELTA
        if action == 10:
            return 5,-5*DELTA
        if action == 11:
            return 6,5*DELTA
        if action == 12:
            return 6,-5*DELTA

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()


class DQN():
	# DQN Agent
	def __init__(self, env):
		# init experience replay
		self.replay_buffer = deque()
		# init some parameters
		self.time_step = 0
		self.epsilon = INITIAL_EPSILON
		self.state_dim = env._get_obs_dim()
		self.action_dim = env._get_action_dim('UR5')

		self.create_Q_network()
		self.create_training_method()

		# Init session
		self.session = tf.InteractiveSession()
		self.session.run(tf.initialize_all_variables())

		# loading networks
		self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state("saved_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
				self.saver.restore(self.session, checkpoint.model_checkpoint_path)
				print("Successfully loaded:", checkpoint.model_checkpoint_path)
		else:
				print("Could not find old network weights")

		global summary_writer
		summary_writer = tf.summary.FileWriter('~/logs',graph=self.session.graph)

	def create_Q_network(self):
		# network weights
		W1 = self.weight_variable([self.state_dim,20])
		b1 = self.bias_variable([20])
		W2 = self.weight_variable([20,self.action_dim])
		b2 = self.bias_variable([self.action_dim])
		# input layer
		self.state_input = tf.placeholder("float",[None,self.state_dim])
        # 数据形状。默认是None，就是一维值，也可以是多维（比如[2,3], [None, 3]表示列是3，行不定）
		# hidden layers
		h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
		# Q Value layer
        # 神经网络输出的是最大的动作的q值
		self.Q_value = tf.matmul(h_layer,W2) + b2


	def create_training_method(self):
		self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot presentation
        # 数据形状。默认是None，就是一维值，也可以是多维（比如[2,3], [None, 3]表示列是3，行不定）
		self.y_input = tf.placeholder("float",[None])
        # 数据形状。默认是None，就是一维值，也可以是多维（比如[2,3], [None, 3]表示列是3，行不定）
		Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices = 1)
		self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
		tf.summary.scalar("loss",self.cost)
		global merged_summary_op
		merged_summary_op = tf.summary.merge_all()
		self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

	def perceive(self,state,action,reward,next_state,done):
		one_hot_action = np.zeros(self.action_dim)
		one_hot_action[action-1] = 1
		self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
		if len(self.replay_buffer) > REPLAY_SIZE:
			self.replay_buffer.popleft()

		if len(self.replay_buffer) > BATCH_SIZE:
			self.train_Q_network()

	def train_Q_network(self):
		self.time_step += 1
		# Step 1: obtain random minibatch from replay memory
		minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		next_state_batch = [data[3] for data in minibatch]

		# Step 2: calculate y
		y_batch = []
		Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
		for i in range(0,BATCH_SIZE):
			done = minibatch[i][4]
			if done:
				y_batch.append(reward_batch[i])
			else :
				y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

		self.optimizer.run(feed_dict={
			self.y_input:y_batch,
			self.action_input:action_batch,
			self.state_input:state_batch
			})
		summary_str = self.session.run(merged_summary_op,feed_dict={
				self.y_input : y_batch,
				self.action_input : action_batch,
				self.state_input : state_batch
				})
		summary_writer.add_summary(summary_str,self.time_step)

		# save network every 1000 iteration
		if self.time_step % 1000 == 0:
			self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step = self.time_step)

	def egreedy_action(self,state):
		Q_value = self.Q_value.eval(feed_dict = {
			self.state_input:[state]
			})[0]
		if random.random() <= self.epsilon:
            #///////////////
			return choice(ACTION_RANGE)
		else:
			return np.argmax(Q_value)
		self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000

	def action(self,state):
		return np.argmax(self.Q_value.eval(feed_dict = {
			self.state_input:[state]
			})[0])

	def weight_variable(self,shape):
		initial = tf.truncated_normal(shape)
		return tf.Variable(initial)

	def bias_variable(self,shape):
		initial = tf.constant(0.01, shape = shape)
		return tf.Variable(initial)


#观察值： 机械手臂关节所处的角度，机械手臂关节的速度，目标物体所处的位置
#动作：机械手臂关节在当前的状态的随机运动a°
#终止态：机械手臂运动出安全区，tip到达target点
EPISODE = 10000 # Episode limitation
STEP = 300 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode

env = ReacherEnv()
agent = DQN(env)

for episode in range(EPISODE):
	# initialize task
	state = env.reset()
	# Train 
	for step in range(STEP):
		action = agent.egreedy_action(state) # e-greedy action for train
		next_state,reward,done= env.step(action)
		# Define reward for agent
		reward_agent = -1 if done else 0.1
		agent.perceive(state,action,reward,next_state,done)
		state = next_state
		if done:
			break

	if episode % 100 == 0:
		total_reward = 0
		for i in range(TEST):
			state = env.reset()
			for j in range(STEP):
				action = agent.action(state) # direct action for test
				state,reward,done = env.step(action)
				total_reward += reward
				if done:
					break
		ave_reward = total_reward/TEST
		print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)
		if ave_reward >= 200:
			break
