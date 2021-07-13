#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 15:13:32 2021

@author: qiang
"""

'''
run this code to start the training.
'''


import tensorflow.keras.backend as K
import tensorflow as tf
from rl_symbol_env_continous import symbol_env_continous
import pandas as pd
from DDPG_agent import ActorCritic

obs_dim = (10,)
obs_num = 10
act_num = 4
act_dim = (4,)

sess = tf.Session()
K.set_session(sess)
env = symbol_env_continous()
actor_critic = ActorCritic(sess)

# If you don't want to load the trained model, comment the following script.
episodee = 3500
actor_critic.actor_model.load_weights(str(episodee)+"_"+"actor"+".h5")
actor_critic.target_actor_model.load_weights(str(episodee)+"_"+"actor_target"+".h5")
actor_critic.critic_model.load_weights(str(episodee)+"_"+"critic"+".h5")
actor_critic.target_critic_model.load_weights(str(episodee)+"_"+"critic_target"+".h5")


EPISODES = 500000
trial_len  = 200
done = 1
history = {"Episode":[],"Success_rate":[]}

max_reward = -999999
global_success_time = 0
for episode in range(EPISODES):
    
    if episode == 0:
        episode = 1
    
    if done == 1:
        cur_state = env.reset()
    if done == 2:
        cur_state = env.reset_o()
    
    print("trial:" + str(episode))
    episode_length = 0
    cur_state = env.reset()
    cur_state = cur_state.reshape((1, obs_num))
    # action = np.random.randint(0,act_dim)
    reward_sum = 0

    for j in range(trial_len):
 			#env.render()
        episode_length += 1
        action = actor_critic.act(cur_state)
        action = action.reshape((1, act_num))
        
        new_state, reward, done= env.step(action)
        
        if done == 1:
            global_success_time += 1
            # print(env.current_dis)
            done_tf = True
        elif done == 2:
            done_tf = True
        elif done == 0:
            done_tf = False
            
        reward_sum += reward
        if j == (trial_len - 1):
            done_tf = True

        actor_critic.train()
        actor_critic.update_target()   
 			
        new_state = new_state.reshape((1,obs_num))
        
        # if episode % 500 ==0:
        #     print("current state ",cur_state)
        #     print("action ",action)
        #     print("reward ",reward)
        #     print("new_state ",new_state)
        #     print("done_tf ",done_tf)


        actor_critic.remember(cur_state, action, reward, new_state, done_tf)
        # print("current state ",cur_state)
        # print("action ",action)
        # print("reward ",reward)
        # print("new_state ",new_state)
        # print("done_tf ",done_tf)
        cur_state = new_state
        if done:
            break
    max_reward = max(reward_sum, max_reward)   
    print('Episode', episode,'| ', 'Episodic Reward', reward_sum,'| ', 'Maximum Reward', max_reward, '| ','End_step',episode_length, '| ','EPSILON', actor_critic.epsilon)
    # print("Success rate: ",global_success_time / episode)
    print('-------------------------------------------------------------------------------------------------------' )
    
    if episode % 500 == 0:
        actor_critic.actor_model.save_weights(str(episode+episodee)+"_"+"actor"+".h5")
        actor_critic.target_actor_model.save_weights(str(episode+episodee)+"_"+"actor_target"+".h5")
        actor_critic.critic_model.save_weights(str(episode+episodee)+"_"+"critic"+".h5")
        actor_critic.target_critic_model.save_weights(str(episode+episodee)+"_"+"critic_target"+".h5")
        
    if episode % 50 == 0:
        print("Success rate: ",global_success_time / 50)
        history["Episode"].append(episode)
        history["Success_rate"].append(global_success_time / 50)
        df = pd.DataFrame(history)
        df.to_csv("history_standard_0.05.csv",index=False,sep=',')
        global_success_time = 0