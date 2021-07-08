#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 23:19:58 2021

@author: qiang
"""

import tensorflow.keras.backend as K
import tensorflow as tf
import time
from rl_symbol_env_continous import symbol_env_continous
from DDPG_keras import ActorCritic
from env_base import UR5_env


obs_dim = (10,)
obs_dimss = 10
act_dim = 4
act_dimension = (4,)


sess = tf.Session()
K.set_session(sess)
env = symbol_env_continous()
actor_critic = ActorCritic(sess)


episodee = 2500
actor_critic.actor_model.load_weights(str(episodee)+"_"+"actor"+".h5")
actor_critic.target_actor_model.load_weights(str(episodee)+"_"+"actor_target"+".h5")
actor_critic.critic_model.load_weights(str(episodee)+"_"+"critic"+".h5")
actor_critic.target_critic_model.load_weights(str(episodee)+"_"+"critic_target"+".h5")

#%%
# Input how many times you want to test the robot.
times = 100
env_rl = symbol_env_continous()
env_sim = UR5_env()
env_sim.sim_start()


for _ in range(times):
    done = False
    cur_state = env_rl.reset()
    el = 0
    target_pos = [cur_state[7],cur_state[8],cur_state[9]]
    env_sim.set_target_pos(target_pos)
    env_sim.movej(env_rl.current_joint_pos)
    
    cur_state = cur_state.reshape((1, obs_dimss))

    while not done and el<=100:
        el += 1
        action = actor_critic.act(cur_state)
        action = action.reshape((1, act_dim))
        next_state, reward, done = env_rl.step(action)
        env_sim.movej(env_rl.current_joint_pos)
        time.sleep(0.05)
        next_state = next_state.reshape((1,obs_dimss))
        cur_state = next_state


        
