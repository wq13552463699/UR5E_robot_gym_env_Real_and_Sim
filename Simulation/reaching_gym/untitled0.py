# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 14:26:45 2021

@author: 14488
"""

from ur5_symbol_v1 import ur5_symbol_v1


env = ur5_symbol_v1()
from stable_baselines3.common.env_checker import check_env
check_env(env)

#%%
import numpy as np
print(np.random.normal(0,2))

#%%

action_low = [-10,-10,-10,-10]
action_high = [10,10,10,10]

print(np.random.uniform(action_low,action_high))

#%%

print(env.action_space.sample())

#%%
step_end_status = 1

a = {'step_end_status':step_end_status}

print(a)

#%%
print(-np.inf)
#%%

a = np.array([1,2,3])
print(a[2])
#%%
print(np.linalg.norm(1))

#%%
def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y

class `StepSignal(gym.GoalEnv)`:

    def __init__(self, input_signal, sample_rate, desired_signal):
        super(StepSignal, self).__init__()

        self.initial_signal = input_signal
        self.signal = self.initial_signal.copy()
        self.sample_rate = sample_rate
        self.desired_signal = desired_signal
        self.distance_threshold = 10e-1

        max_offset = abs(max( max(self.desired_signal) , max(self.signal))
                 - min( min(self.desired_signal) , min(self.signal)) )

        self.action_space = spaces.Box(low=np.array([10e-4,-max_offset]),\
high=np.array([self.sample_rate/2-0.1,max_offset]), dtype=np.float16)

        obs = self._get_obs()
        self.observation_space = spaces.Dict(dict(
        desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
        achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
        observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    def step(self, action):
        range = self.action_space.high - self.action_space.low
        action = range / 2 * (action + 1)
        self._set_action(action)
        obs = self._get_obs()
        done = False

        info = {
                'is_success': self._is_success(obs['achieved_goal'], self.desired_signal),
               }
        reward = -self.compute_reward(obs['achieved_goal'],self.desired_signal)
        return obs, reward, done, info

    def reset(self):
        self.signal = self.initial_signal.copy()
        return self._get_obs()


    def _set_action(self, actions):
        actions = np.clip(actions,a_max=self.action_space.high,a_min=self.action_space.low)
        cutoff = actions[0]
        offset = actions[1]
        print(cutoff, offset)
        self.signal = butter_lowpass_filter(self.signal, cutoff, self.sample_rate/2) - offset

    def _get_obs(self):
        obs = self.signal
        achieved_goal = self.signal
        return {
        'observation': obs.copy(),
        'achieved_goal': achieved_goal.copy(),
        'desired_goal': self.desired_signal.copy(),
        }

    def compute_reward(self, goal_achieved, goal_desired):
        d = np.linalg.norm(goal_desired-goal_achieved)
        return d


    def _is_success(self, achieved_goal, desired_goal):
        d = self.compute_reward(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)